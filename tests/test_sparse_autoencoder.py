import torch
import torch.nn as nn

from modellens.core.lens import ModelLens
from modellens.analysis.sparse_autoencoder import train_sae


def _tiny_mlp(in_dim=8, hidden=8, num_classes=5):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),   # "0" -> hidden (SAE hook site)
        nn.ReLU(),
        nn.Linear(hidden, hidden),   # "2"
        nn.ReLU(),
        nn.Linear(hidden, num_classes),  # "4"
    )


class _ResidualMLP(nn.Module):
    """
    A feedforward net with a submodule named 'downsample' so the adapter
    reports a residual stream — used to exercise the residual-layer default.
    """

    def __init__(self, in_dim=8, hidden=8, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.downsample = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = torch.relu(self.fc1(x)) + self.downsample(x)
        return self.fc2(h)


def test_train_sae_recon_decreases():
    torch.manual_seed(0)
    model = _tiny_mlp()
    lens = ModelLens(model)

    inputs = torch.randn(32, 8)
    sae, summary = train_sae(
        lens, inputs, layer_name="0", expansion=4, steps=150, lr=1e-2, seed=0
    )

    # (a) reconstruction loss decreases with training
    assert summary["final_recon_loss"] < summary["initial_recon_loss"]

    # Overcomplete hidden dim and documented summary fields
    assert summary["num_features"] == 8 * 4
    assert summary["layer_name"] == "0"
    for key in ("l0_sparsity", "l1_sparsity", "dead_features"):
        assert key in summary
    lens.clear()


def test_sae_features_shape_and_nonneg():
    torch.manual_seed(0)
    model = _tiny_mlp()
    lens = ModelLens(model)

    train_inputs = torch.randn(32, 8)
    sae, _ = train_sae(lens, train_inputs, layer_name="0", steps=100, lr=1e-2)

    eval_inputs = [torch.randn(1, 8) for _ in range(4)]
    top_k = 3
    results = lens.sae_features(eval_inputs, sae, layer_name="0", top_k=top_k)

    # (b) documented return shape
    expected_keys = {
        "layer_name",
        "per_input",
        "feature_summary",
        "num_features",
        "num_inputs",
        "num_active_features",
    }
    assert expected_keys.issubset(results.keys())
    assert results["num_inputs"] == len(eval_inputs)
    assert len(results["per_input"]) == len(eval_inputs)

    for entry in results["per_input"]:
        assert len(entry["top_features"]) == min(top_k, sae.num_features)
        # (c) feature activations are non-negative (ReLU code)
        assert torch.all(entry["activations"] >= 0)
        for _, value in entry["top_features"]:
            assert value >= 0

    # Per-feature inspectability: top activating inputs/tokens.
    for feat, data in results["feature_summary"].items():
        assert len(data["top_activations"]) <= top_k
        for hit in data["top_activations"]:
            assert hit["value"] >= 0
            assert 0 <= hit["input_index"] < len(eval_inputs)
    lens.clear()


def test_residual_layer_default_resolves():
    torch.manual_seed(0)
    model = _ResidualMLP()
    lens = ModelLens(model)

    inputs = torch.randn(16, 8)
    # No layer_name -> should default to a residual-stream layer.
    sae, summary = train_sae(lens, inputs, steps=20, lr=1e-2)

    assert summary["layer_name"] is not None
    assert summary["final_recon_loss"] <= summary["initial_recon_loss"]
    lens.clear()
