import pytest
import torch
import torch.nn as nn
from modellens.core import ModelLens
from modellens.adapters.base import AnalysisCapability, UnsupportedAnalysisError


@pytest.fixture(scope="session")
def simple_cnn():
    """A basic CNN for convolutional tests."""

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(32, 10)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.flatten(1)
            return self.classifier(x)

    return CNN()


# Test - Detect Architectures
class TestAutoDetection:
    """Verify ModelLens correctly identifies model architectures."""

    def test_detects_cnn(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        assert lens.adapter.type_of_adapter == "pytorch"
        assert lens.adapter.architecture_family == "convolutional"


# Test - Detect model capabilities
class TestCapabilities:
    """Verify each adapter declares the right capabilities."""

    def test_cnn_capabilities(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        caps = lens.adapter.capabilities()

        assert AnalysisCapability.HOOKS in caps
        assert AnalysisCapability.FILTER_ANALYSIS in caps
        assert AnalysisCapability.FEATURE_MAP_ANALYSIS in caps
        assert AnalysisCapability.ACTIVATION_PATCHING in caps
        # CNN should NOT have these
        assert AnalysisCapability.ATTENTION_MAPS not in caps
        assert AnalysisCapability.GATE_ANALYSIS not in caps


# Test - Detect unsupported analysis
class TestCapabilityGating:
    """Verify unsupported analyses raise clear errors."""

    def test_attention_on_cnn_raises(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        with pytest.raises(UnsupportedAnalysisError, match="attention_map"):
            lens.attention_map(torch.randn(1, 1, 28, 28))

    def test_error_message_lists_available(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        with pytest.raises(UnsupportedAnalysisError) as exc_info:
            lens.attention_map(torch.randn(1, 1, 28, 28))
        error_msg = str(exc_info.value)
        assert "filter_analysis" in error_msg
        assert "convolutional" in error_msg


# Test - Check summary output
class TestSummary:
    """Verify summary output is correct and complete."""

    def test_cnn_summary(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        s = lens.summary()

        assert s["architecture"] == "convolutional"
        assert "filter_analysis" in s["available_analyses"]
        assert "attention_maps" in s["unavailable_analyses"]


# Test - Hook Management
class TestHooks:
    """Verify hooks work correctly across architectures."""

    def test_attach_and_capture_cnn(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        lens.attach_layers(["conv1", "conv2"])

        lens.run(torch.randn(1, 1, 28, 28))

        activations = lens.get_activations()
        assert "conv1" in activations
        assert activations["conv1"].shape[1] == 16  # 16 filters


# Test - Patchable Layers
class TestPatchableLayers:
    """Verify each adapter returns correct patchable layers."""

    def test_cnn_patchable(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        patchable = lens.patchable_layers()

        assert "conv1" in patchable
        assert "conv2" in patchable
        assert "classifier" in patchable


# Test - Sequential Layers
class TestSequentialLayers:
    """Verify execution order is correct."""

    def test_cnn_order(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        seq = lens.adapter.get_sequential_layers()

        assert len(seq) > 0
        # conv1 should come before conv2
        conv1_idx = seq.index("conv1")
        conv2_idx = seq.index("conv2")
        assert conv1_idx < conv2_idx


# Test - Output Projection
class TestOutputProjection:
    """Verify output projection detection works."""

    def test_cnn_has_projection(self, simple_cnn):
        lens = ModelLens(simple_cnn)
        proj = lens.adapter.get_output_projection()

        assert proj is not None
        # Classifier has 10 classes
        assert proj.shape[0] == 10
