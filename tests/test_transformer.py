import pytest
import torch
import torch.nn as nn
from modellens.core import ModelLens
from modellens.adapters.base import AnalysisCapability, UnsupportedAnalysisError


@pytest.fixture(scope="session")
def gpt2():
    """Load GPT-2 small once for all transformer tests."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


# Test - Detect Architectures
class TestAutoDetection:
    """Verify ModelLens correctly identifies model architectures."""

    def test_detects_huggingface(self, gpt2):
        model, _ = gpt2
        lens = ModelLens(model)
        assert lens.adapter.type_of_adapter == "huggingface"
        assert lens.adapter.architecture_family == "transformer"


# Test - Detect model capabilities
class TestCapabilities:
    """Verify each adapter declares the right capabilities."""

    def test_transformer_capabilities(self, gpt2):
        model, _ = gpt2
        lens = ModelLens(model)
        caps = lens.adapter.capabilities()

        assert AnalysisCapability.HOOKS in caps
        assert AnalysisCapability.ATTENTION_MAPS in caps
        assert AnalysisCapability.LAYER_PROBING in caps
        assert AnalysisCapability.ACTIVATION_PATCHING in caps
        assert AnalysisCapability.EMBEDDINGS in caps
        assert AnalysisCapability.RESIDUAL_STREAM in caps
        # Transformer should NOT have these
        assert AnalysisCapability.GATE_ANALYSIS not in caps
        assert AnalysisCapability.FILTER_ANALYSIS not in caps

    def test_supports_method(self, gpt2):
        model, _ = gpt2
        lens = ModelLens(model)
        assert lens.adapter.supports(AnalysisCapability.ATTENTION_MAPS) is True
        assert lens.adapter.supports(AnalysisCapability.GATE_ANALYSIS) is False


# Test - Detect unsupported analysis
class TestCapabilityGating:
    def test_gate_on_transformer_raises(self, gpt2):
        model, _ = gpt2
        lens = ModelLens(model)
        with pytest.raises(UnsupportedAnalysisError, match="gate_analysis"):
            lens.gate_analysis("test input")


# Test - Check summary output
class TestSummary:
    """Verify summary output is correct and complete."""

    def test_transformer_summary(self, gpt2):
        model, _ = gpt2
        lens = ModelLens(model)
        s = lens.summary()

        assert s["backend"] == "huggingface"
        assert s["architecture"] == "transformer"
        assert s["total_parameters"] > 0
        assert s["layer_count"] > 0
        assert "attention_maps" in s["available_analyses"]
        assert "gate_analysis" in s["unavailable_analyses"]


# Test - Hook Management
class TestHooks:
    """Verify hooks work correctly across architectures."""

    def test_attach_and_capture_transformer(self, gpt2):
        model, tokenizer = gpt2
        lens = ModelLens(model)
        lens.adapter.set_tokenizer(tokenizer)

        layers = lens.adapter.get_sequential_layers()[:3]
        lens.attach_layers(layers)

        inputs = tokenizer("Hello world", return_tensors="pt")
        lens.run(inputs)

        activations = lens.get_activations()
        assert len(activations) == 3
        for name in layers:
            assert name in activations
            assert isinstance(activations[name], torch.Tensor)


# Test - Patchable Layers
class TestPatchableLayers:
    """Verify each adapter returns correct patchable layers."""

    def test_transformer_patchable(self, gpt2):
        model, _ = gpt2
        lens = ModelLens(model)
        patchable = lens.patchable_layers()

        assert len(patchable) > 0
        # GPT-2 should have attn and mlp sublayers
        has_attn = any("attn" in name for name in patchable)
        has_mlp = any("mlp" in name for name in patchable)
        assert has_attn, f"No attn layers found in: {patchable}"
        assert has_mlp, f"No mlp layers found in: {patchable}"


# Test - Sequential Layers
class TestSequentialLayers:
    """Verify execution order is correct."""

    def test_transformer_order(self, gpt2):
        model, _ = gpt2
        lens = ModelLens(model)
        seq = lens.adapter.get_sequential_layers()

        assert len(seq) == 12  # GPT-2 small has 12 blocks
        # Should be in order
        for i, name in enumerate(seq):
            assert f".{i}" in name, f"Layer {name} doesn't match index {i}"


# Test - Output Projection
class TestOutputProjection:
    """Verify output projection detection works."""

    def test_backward_compat_unembedding(self, gpt2):
        """get_unembedding() should still work as alias."""
        model, _ = gpt2
        lens = ModelLens(model)
        proj = lens.adapter.get_unembedding()
        assert proj is not None
        assert proj.shape[0] == 50257
