"""
Qwen3 support in the HuggingFace adapter.

Before "qwen3" was in _ARCHITECTURE_PATTERNS, layer lookup fell back to keyword
search and returned sub-modules (q_proj, q_norm, ...) as if they were layers.
A wrapped model (PEFT prefixes names with "base_model.model.") must fail loudly
instead of silently returning no layers.
"""

import pytest
import torch
import torch.nn as nn

from modellens.adapters.huggingface_adapter import HuggingFaceAdapter
from modellens.core.lens import ModelLens

N_LAYERS = 2


# ---- Fixtures ----


@pytest.fixture(scope="module")
def tiny_qwen3():
    """A tiny randomly-initialized Qwen3: no download, real HF architecture."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
    )
    return Qwen3ForCausalLM(config).eval()


class _PeftStyleWrapper(nn.Module):
    """Reproduces PEFT's module naming: base_model.model.<inner names>."""

    def __init__(self, inner):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = inner
        self.config = inner.config


# ---- Layer detection ----


def test_qwen3_dispatches_to_huggingface_adapter(tiny_qwen3):
    lens = ModelLens(tiny_qwen3)
    assert isinstance(lens.adapter, HuggingFaceAdapter)
    assert lens.adapter._model_type == "qwen3"


def test_qwen3_patchable_layers_are_blocks_not_submodules(tiny_qwen3):
    layers = HuggingFaceAdapter(tiny_qwen3).get_patchable_layers()
    expected = []
    for i in range(N_LAYERS):
        expected += [f"model.layers.{i}.self_attn", f"model.layers.{i}.mlp"]
    assert layers == expected
    assert not any(name.endswith(("q_proj", "q_norm", "k_norm")) for name in layers)


def test_qwen3_sequential_and_attention_layers(tiny_qwen3):
    adapter = HuggingFaceAdapter(tiny_qwen3)
    assert adapter.get_sequential_layers() == [f"model.layers.{i}" for i in range(N_LAYERS)]
    assert adapter.get_attention_layers() == [
        f"model.layers.{i}.self_attn" for i in range(N_LAYERS)
    ]


# ---- Wrapped models fail loudly ----


@pytest.mark.parametrize(
    "method", ["get_patchable_layers", "get_sequential_layers", "get_attention_layers"]
)
def test_wrapped_model_raises_instead_of_returning_no_layers(tiny_qwen3, method):
    adapter = HuggingFaceAdapter(_PeftStyleWrapper(tiny_qwen3))
    with pytest.raises(ValueError, match="merge_and_unload"):
        getattr(adapter, method)()


def test_real_peft_model_raises_and_merged_model_works(tiny_qwen3):
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"])
    wrapped = peft.get_peft_model(tiny_qwen3, config)

    with pytest.raises(ValueError, match="merge_and_unload"):
        HuggingFaceAdapter(wrapped).get_patchable_layers()

    merged = wrapped.merge_and_unload()
    assert len(HuggingFaceAdapter(merged).get_patchable_layers()) == 2 * N_LAYERS
