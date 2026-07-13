import torch
import torch.nn as nn

from modellens.core.lens import ModelLens
from modellens.analysis.probing import train_probe, probe_sweep


def _mlp(in_dim=8, hidden=16, num_classes=2):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),   # "0" — probe target (pre-ReLU, linear in x)
        nn.ReLU(),
        nn.Linear(hidden, num_classes),  # "2"
    )


def _linearly_separable(n=200):
    """label = (first feature > 0) — linearly decodable from any linear map of x."""
    X = torch.randn(n, 8)
    labels = (X[:, 0] > 0).long().tolist()
    return X, labels


def test_train_probe_learns_linear_concept():
    torch.manual_seed(0)
    lens = ModelLens(_mlp())
    X, labels = _linearly_separable()

    probe, summary = train_probe(lens, "0", X, labels, steps=300, lr=0.05)

    assert summary["num_classes"] == 2
    assert summary["input_dim"] == 16          # pooled dim at layer "0"
    assert summary["num_examples"] == 200
    assert summary["num_test"] > 0
    # The concept is linearly decodable, and it generalizes above chance.
    assert summary["train_accuracy"] > 0.9
    assert summary["test_accuracy"] > 0.8
    assert summary["test_accuracy"] > summary["baseline"]


def test_probe_sweep_reports_per_layer_and_best():
    torch.manual_seed(0)
    lens = ModelLens(_mlp())
    X, labels = _linearly_separable()

    result = probe_sweep(lens, X, labels, steps=200, lr=0.05)

    assert result["num_layers_probed"] >= 1
    assert result["best_layer"] is not None
    assert result["best_accuracy"] > 0.8
    # layer_scores maps each probed layer to a held-out accuracy
    assert set(result["layer_scores"].keys()) == {
        p["layer_name"] for p in result["per_layer"]
    }


def test_apply_probe_returns_labelled_predictions():
    torch.manual_seed(0)
    lens = ModelLens(_mlp())
    X = torch.randn(120, 8)
    labels = ["pos" if v > 0 else "neg" for v in X[:, 0]]

    probe, _ = train_probe(lens, "0", X, labels, steps=200, lr=0.05)

    eval_X = torch.randn(10, 8)
    out = lens.apply_probe(eval_X, probe, "0")

    assert out["num_inputs"] == 10
    assert len(out["predictions"]) == 10
    # predictions are mapped back to the original string labels
    assert set(out["predictions"]).issubset({"pos", "neg"})
    assert out["probabilities"].shape == (10, 2)
    assert torch.allclose(
        out["probabilities"].sum(dim=-1), torch.ones(10), atol=1e-4
    )


def test_probe_list_inputs_align_with_labels():
    """List-of-examples path (variable-length inputs) pools one vector each."""
    torch.manual_seed(0)
    lens = ModelLens(_mlp())
    inputs = [torch.randn(1, 8) for _ in range(30)]
    labels = [int(x[0, 0] > 0) for x in inputs]

    probe, summary = train_probe(lens, "0", inputs, labels, steps=150, lr=0.05)
    assert summary["num_examples"] == 30
    assert summary["train_accuracy"] > 0.8
