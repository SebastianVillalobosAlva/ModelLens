"""
Attribution patching — gradient approximation of activation patching.

Key correctness guarantee: for a *linear* network with a *linear* metric, the
first-order approximation is exact, so attribution effects must equal the exact
patching effects. The scaling win (constant forward passes vs. one-per-layer)
is checked by counting model invocations.
"""

import pytest
import torch
import torch.nn as nn

from modellens.core.lens import ModelLens
from modellens.analysis.activation_patching import (
    run_activation_patching,
    run_attribution_patching,
)


@pytest.fixture(scope="module")
def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=2)
    return ModelLens(GPT2LMHeadModel(config).eval())


def _ids(seq):
    return {"input_ids": torch.tensor([seq])}


# ---- Correctness: exact on a linear model + linear metric ----


def test_attribution_matches_exact_on_linear_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 6), nn.Linear(6, 4))  # purely linear
    lens = ModelLens(model)

    clean = torch.randn(1, 6)
    corrupted = torch.randn(1, 6)
    layers = ["0", "1"]

    exact = run_activation_patching(
        lens, clean, corrupted, layer_names=layers,
        metric_fn=lambda o: o[:, 0].sum().item(),
    )
    attr = run_attribution_patching(
        lens, clean, corrupted, layer_names=layers,
        metric_fn=lambda o: o[:, 0].sum(),
    )

    for name in layers:
        assert attr["patch_effects"][name]["attribution"] == pytest.approx(
            exact["patch_effects"][name]["effect"], abs=1e-4
        )


# ---- Scaling: constant forward passes, independent of layer count ----


def test_attribution_uses_constant_forward_passes(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    patchable = tiny_gpt2.adapter.get_patchable_layers()
    assert len(patchable) > 2  # so the contrast below is meaningful

    def count_forwards(fn):
        counter = {"n": 0}
        handle = tiny_gpt2.model.register_forward_hook(
            lambda *_: counter.__setitem__("n", counter["n"] + 1)
        )
        try:
            fn()
        finally:
            handle.remove()
        return counter["n"]

    attr_passes = count_forwards(
        lambda: run_attribution_patching(tiny_gpt2, clean, corrupted)
    )
    exact_passes = count_forwards(
        lambda: run_activation_patching(tiny_gpt2, clean, corrupted)
    )

    # Attribution: 1 corrupted + 1 clean pass, regardless of layer count.
    assert attr_passes == 2
    # Exact: clean + corrupted + one pass per patched layer.
    assert exact_passes == 2 + len(patchable)


# ---- Schema + lens method on a transformer ----


def test_attribution_patch_schema_on_transformer(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    result = tiny_gpt2.attribution_patch(clean, corrupted)

    assert result["method"] == "attribution"
    assert set(result) >= {"clean_metric", "corrupted_metric", "total_effect", "patch_effects"}
    assert len(result["patch_effects"]) == len(tiny_gpt2.adapter.get_patchable_layers())
    for data in result["patch_effects"].values():
        for field in ("attribution", "effect", "normalized_effect", "patched_metric"):
            assert field in data


def test_attribution_metric_must_return_tensor(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    with pytest.raises(ValueError):
        # float metric is the exact-patching contract, not attribution's
        run_attribution_patching(
            tiny_gpt2, clean, corrupted, metric_fn=lambda o: o.logits[:, -1].max().item()
        )


# ---- discover_circuit with the attribution backend ----


def test_discover_circuit_attribution_method(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    circuit = tiny_gpt2.discover_circuit(
        clean, corrupted, importance_threshold=0.0, method="attribution"
    )

    assert len(circuit["nodes"]) >= 1
    valid_roles = {"critical", "booster", "gate", "processor"}
    for node in circuit["nodes"]:
        assert node["role"] in valid_roles


def test_discover_circuit_rejects_unknown_method(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    with pytest.raises(ValueError):
        tiny_gpt2.discover_circuit(clean, corrupted, method="bogus")
