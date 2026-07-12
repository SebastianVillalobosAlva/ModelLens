"""
Regression coverage for the core hooks / patching path that circuit discovery
runs through. The existing tests cover the newer modules (SAE, MCP, layer
evolution); these guard the machinery underneath discover_circuit, which a
live downstream consumer (Stoic-Steering circuit sweep) depends on.
"""

import pytest
import torch
import torch.nn as nn

from modellens.core.lens import ModelLens
from modellens.adapters.pytorch_adapter import PyTorchAdapter
from modellens.adapters.huggingface_adapter import HuggingFaceAdapter
from modellens.analysis.activation_patching import (
    run_activation_patching,
    _capture_activations,
    _default_metric,
)


# ---- Fixtures ----


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture()
def tiny_net():
    torch.manual_seed(0)
    return _TinyNet().eval()


@pytest.fixture(scope="module")
def tiny_gpt2():
    """A tiny randomly-initialized GPT-2 — no download, real HF architecture."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=2
    )
    model = GPT2LMHeadModel(config).eval()
    return ModelLens(model)


def _ids(seq):
    return {"input_ids": torch.tensor([seq])}


# ---- 1. Hook cleanup / _capture_activations (guards the 44d9b77 bug) ----


def test_capture_activations_returns_correct_activations(tiny_net):
    inputs = torch.randn(2, 4)
    available = dict(tiny_net.named_modules())

    # If make_hook regressed to returning None (the 44d9b77 bug),
    # register_forward_hook(None) would raise here.
    activations, output = _capture_activations(
        tiny_net, available, inputs, ["fc1", "fc2"]
    )

    assert set(activations.keys()) == {"fc1", "fc2"}
    assert activations["fc1"].shape == (2, 4)
    assert activations["fc2"].shape == (2, 3)


def test_capture_hooks_do_not_alter_forward_output(tiny_net):
    inputs = torch.randn(2, 4)
    available = dict(tiny_net.named_modules())

    clean_output = tiny_net(inputs)  # plain forward, no hooks
    _, captured_output = _capture_activations(
        tiny_net, available, inputs, ["fc1", "fc2"]
    )

    # A forward hook returning non-None silently replaces the output;
    # the capture hook must return None and leave the output untouched.
    assert torch.allclose(clean_output, captured_output)


def test_capture_removes_all_hooks_on_exit(tiny_net):
    inputs = torch.randn(2, 4)
    available = dict(tiny_net.named_modules())

    _capture_activations(tiny_net, available, inputs, ["fc1", "fc2"])

    for module in tiny_net.modules():
        assert len(module._forward_hooks) == 0


def test_patching_leaves_no_lingering_hooks(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    run_activation_patching(tiny_gpt2, clean, corrupted)

    # Every patch scopes and removes its own hook; nothing should linger.
    for module in tiny_gpt2.model.modules():
        assert len(module._forward_hooks) == 0


# ---- 2. Custom metric_fn dispatch ----


def test_custom_metric_fn_is_used_not_default(tiny_gpt2):
    calls = []

    def my_metric(output):
        calls.append(1)
        return 42.0

    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    result = run_activation_patching(
        tiny_gpt2, clean, corrupted, metric_fn=my_metric
    )

    # The user metric was actually invoked (clean + corrupted + per patch)...
    assert len(calls) >= 2
    # ...and never silently replaced by _default_metric (max-logit), which
    # would not return the sentinel 42.0.
    assert result["clean_metric"] == 42.0
    assert result["corrupted_metric"] == 42.0


def test_default_metric_used_when_none(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    result = run_activation_patching(tiny_gpt2, clean, corrupted)

    with torch.no_grad():
        expected = _default_metric(tiny_gpt2.model(**clean))
    assert result["clean_metric"] == pytest.approx(expected)


# ---- 3. Adapter dispatch ----


def test_pytorch_adapter_dispatch_and_patchable_layers(tiny_net):
    lens = ModelLens(tiny_net)
    assert isinstance(lens.adapter, PyTorchAdapter)
    assert lens.adapter.architecture_family == "feedforward"

    patchable = lens.adapter.get_patchable_layers()
    assert set(patchable) == {"fc1", "fc2"}  # every Linear, nothing else


def test_huggingface_adapter_dispatch_and_patchable_layers(tiny_gpt2):
    assert isinstance(tiny_gpt2.adapter, HuggingFaceAdapter)

    patchable = tiny_gpt2.adapter.get_patchable_layers()
    assert len(patchable) > 0
    assert any("attn" in n for n in patchable)
    assert any("mlp" in n for n in patchable)


# ---- 4. Circuit discovery end-to-end ----


def test_discover_circuit_wellformed_nodes(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    circuit = tiny_gpt2.discover_circuit(clean, corrupted, importance_threshold=0.0)

    assert len(circuit["nodes"]) >= 1
    assert circuit["num_components"] == len(circuit["nodes"])
    valid_roles = {"critical", "booster", "gate", "processor"}
    for node in circuit["nodes"]:
        for field in ("name", "family", "normalized_effect", "role"):
            assert field in node
        assert node["role"] in valid_roles


def test_discover_circuit_respects_importance_threshold(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    threshold = 0.5
    circuit = tiny_gpt2.discover_circuit(
        clean, corrupted, importance_threshold=threshold
    )
    for node in circuit["nodes"]:
        assert abs(node["normalized_effect"]) >= threshold


def test_discover_circuit_empty_path_returns_cleanly(tiny_gpt2):
    clean, corrupted = _ids([1, 2, 3, 4, 5]), _ids([1, 2, 9, 4, 5])
    # No component can exceed an astronomically high threshold.
    circuit = tiny_gpt2.discover_circuit(
        clean, corrupted, importance_threshold=1e9
    )
    assert circuit["nodes"] == []
    assert circuit["edges"] == []
    assert "message" in circuit


# ---- 5. Logit-lens shapes ----


def test_logit_lens_per_layer_shapes(tiny_gpt2):
    top_k = 3
    results = tiny_gpt2.layer_probe(_ids([1, 2, 3, 4, 5]), top_k=top_k)

    layer_results = results["layer_results"]
    assert len(layer_results) >= 1
    for data in layer_results.values():
        assert data["top_k_indices"].shape[-1] == top_k
        assert data["top_k_probs"].shape == data["top_k_indices"].shape
        # probs are a valid distribution over the vocab
        assert torch.allclose(
            data["probs"].sum(dim=-1),
            torch.ones_like(data["probs"].sum(dim=-1)),
            atol=1e-4,
        )
