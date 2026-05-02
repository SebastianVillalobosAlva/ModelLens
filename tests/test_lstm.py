"""
tests/test_core_redesign.py

Validates the adapter redesign works end-to-end.
Run with: pytest tests/test_core_redesign.py -v
"""

import pytest
import torch
import torch.nn as nn
from modellens.core import ModelLens
from modellens.adapters.base import AnalysisCapability, UnsupportedAnalysisError


@pytest.fixture(scope="session")
def simple_lstm():
    """A basic LSTM for recurrent tests."""

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 32)
            self.lstm = nn.LSTM(32, 64, batch_first=True, num_layers=2)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return LSTMModel()


# Test - Detect Architectures
class TestAutoDetection:
    """Verify ModelLens correctly identifies model architectures."""

    def test_detects_lstm(self, simple_lstm):
        lens = ModelLens(simple_lstm)
        assert lens.adapter.type_of_adapter == "pytorch"
        assert lens.adapter.architecture_family == "recurrent"


# Test - Detect model capabilities
class TestCapabilities:
    """Verify each adapter declares the right capabilities."""

    def test_lstm_capabilities(self, simple_lstm):
        lens = ModelLens(simple_lstm)
        caps = lens.adapter.capabilities()

        assert AnalysisCapability.HOOKS in caps
        assert AnalysisCapability.GATE_ANALYSIS in caps
        assert AnalysisCapability.ACTIVATION_PATCHING in caps
        assert AnalysisCapability.EMBEDDINGS in caps
        # LSTM should NOT have these
        assert AnalysisCapability.ATTENTION_MAPS not in caps
        assert AnalysisCapability.FILTER_ANALYSIS not in caps


# Test - Detect unsupported analysis
class TestCapabilityGating:
    """Verify unsupported analyses raise clear errors."""

    def test_attention_on_lstm_raises(self, simple_lstm):
        lens = ModelLens(simple_lstm)
        with pytest.raises(UnsupportedAnalysisError, match="attention_map"):
            lens.attention_map(torch.randint(0, 100, (1, 10)))


# Test - Hook Management
class TestHooks:
    """Verify hooks work correctly across architectures."""

    def test_attach_and_capture_lstm(self, simple_lstm):
        lens = ModelLens(simple_lstm)
        lens.attach_layers(["embedding", "lstm"])

        lens.run(torch.randint(0, 100, (1, 10)))

        activations = lens.get_activations()
        assert "embedding" in activations
        assert "lstm" in activations


# Test - Patchable Layers
class TestPatchableLayers:
    """Verify each adapter returns correct patchable layers."""

    def test_lstm_patchable(self, simple_lstm):
        lens = ModelLens(simple_lstm)
        patchable = lens.patchable_layers()

        assert "lstm" in patchable
        assert "fc" in patchable
