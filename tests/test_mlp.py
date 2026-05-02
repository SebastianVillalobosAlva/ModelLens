import pytest
import torch
import torch.nn as nn
from modellens.core import ModelLens
from modellens.adapters.base import AnalysisCapability, UnsupportedAnalysisError


@pytest.fixture(scope="session")
def simple_mlp():
    """A basic feedforward network for MLP tests."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(32, 16)
            self.relu2 = nn.ReLU()
            self.output = nn.Linear(16, 5)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            return self.output(x)

    return MLP()


# Test - Detect Architectures
class TestAutoDetection:
    """Verify ModelLens correctly identifies model architectures."""

    def test_detects_mlp(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        assert lens.adapter.type_of_adapter == "pytorch"
        assert lens.adapter.architecture_family == "feedforward"

    def test_manual_backend_override(self, simple_mlp):
        lens = ModelLens(simple_mlp, backend="pytorch")
        assert lens.adapter.type_of_adapter == "pytorch"

    def test_invalid_backend_raises(self, simple_mlp):
        with pytest.raises(ValueError, match="Unknown backend"):
            ModelLens(simple_mlp, backend="nonexistent")


# Test - Detect model capabilities
class TestCapabilities:
    """Verify each adapter declares the right capabilities."""

    def test_mlp_capabilities(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        caps = lens.adapter.capabilities()

        assert AnalysisCapability.HOOKS in caps
        assert AnalysisCapability.ACTIVATION_PATCHING in caps
        assert AnalysisCapability.LAYER_PROBING in caps
        # MLP should NOT have these
        assert AnalysisCapability.ATTENTION_MAPS not in caps
        assert AnalysisCapability.GATE_ANALYSIS not in caps
        assert AnalysisCapability.FILTER_ANALYSIS not in caps
        assert AnalysisCapability.EMBEDDINGS not in caps


# Test - Detect unsupported analysis
class TestCapabilityGating:
    """Verify unsupported analyses raise clear errors."""

    def test_filter_on_mlp_raises(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        with pytest.raises(UnsupportedAnalysisError, match="filter_analysis"):
            lens.filter_analysis(torch.randn(1, 10))


# Test - Check summary output
class TestSummary:
    """Verify summary output is correct and complete."""

    def test_repr(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        r = repr(lens)
        assert "feedforward" in r
        assert "pytorch" in r


# Test - Hook Management
class TestHooks:
    """Verify hooks work correctly across architectures."""

    def test_attach_and_capture_mlp(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        lens.attach_layers(["fc1", "fc2"])

        lens.run(torch.randn(1, 10))

        activations = lens.get_activations()
        assert "fc1" in activations
        assert "fc2" in activations

    def test_clear_hooks(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        lens.attach_all()
        assert len(lens.hooks) > 0

        lens.clear()
        assert len(lens.hooks) == 0

    def test_chaining(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        result = lens.attach_layers(["fc1", "fc2"])
        assert result is lens  # Returns self for chaining


# Test - Patchable Layers
class TestPatchableLayers:
    """Verify each adapter returns correct patchable layers."""

    def test_mlp_patchable(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        patchable = lens.patchable_layers()

        assert "fc1" in patchable
        assert "fc2" in patchable
        assert "output" in patchable


# Test - Output Projection
class TestOutputProjection:
    """Verify output projection detection works."""

    def test_mlp_has_projection(self, simple_mlp):
        lens = ModelLens(simple_mlp)
        proj = lens.adapter.get_output_projection()

        assert proj is not None
        # Output layer has 5 classes
        assert proj.shape[0] == 5
