import torch
import torch.nn as nn

from modellens.core.lens import ModelLens


def _tiny_mlp(in_dim=8, hidden=6, num_classes=5):
    """
    A tiny feedforward net whose hidden widths all match the output
    projection's input dim, so several layers are projectable through
    the (hidden -> classes) head.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden),   # "0"  -> hidden
        nn.ReLU(),
        nn.Linear(hidden, hidden),   # "2"  -> hidden
        nn.ReLU(),
        nn.Linear(hidden, hidden),   # "4"  -> hidden
        nn.ReLU(),
        nn.Linear(hidden, num_classes),  # "6" -> output projection
    )


def test_layer_evolution_smoke():
    torch.manual_seed(0)
    num_classes = 5
    top_k = 3
    model = _tiny_mlp(num_classes=num_classes)
    lens = ModelLens(model)

    inputs = torch.randn(1, 8)
    results = lens.layer_evolution(inputs, top_k=top_k)

    # Documented return-dict shape (see run_layer_evolution)
    expected_keys = {
        "layers",
        "layers_ordered",
        "entropy_trajectory",
        "confidence_trajectory",
        "kl_trajectory",
        "margin_trajectory",
        "token_trajectories",
        "key_moments",
        "num_layers",
        "position_used",
    }
    assert expected_keys.issubset(results.keys())

    # At least one projectable layer was tracked.
    assert results["num_layers"] >= 1
    assert len(results["layers"]) == results["num_layers"]
    assert len(results["layers_ordered"]) == results["num_layers"]
    assert len(results["entropy_trajectory"]) == results["num_layers"]

    # Per-layer shapes: top-k distribution over the vocab/class space.
    for layer in results["layers"]:
        assert layer["top_k_indices"].shape == (top_k,)
        assert layer["top_k_probs"].shape == (top_k,)
        # Probabilities are valid.
        assert torch.all(layer["top_k_probs"] >= 0)
        assert 0.0 <= layer["top1_prob"] <= 1.0

    lens.clear()
