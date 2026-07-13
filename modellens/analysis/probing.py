"""
Linear concept probing.

Train a small linear classifier on the activations ModelLens captures at a
layer to ask *what a representation exposes* about a concept — and sweep across
layers to find *where* that concept is most linearly decodable.

Training a probe fits a new classifier, so `train_probe` / `probe_sweep` are
module-level functions (like `train_sae`), not lens methods — the lens is for
inspection. Running a trained probe (`apply_probe`) is inspection and is also
exposed as `ModelLens.apply_probe`.

Probes only need activation capture (the HOOKS capability), so they work on any
architecture. A concept the probe can't read out linearly is not necessarily
absent — it may just be non-linearly encoded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from modellens.adapters.base import AnalysisCapability


class LinearProbe(nn.Module):
    """A single linear layer trained to read a concept off frozen activations."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)
        # Populated by train_probe so predictions map back to original labels.
        self.classes_: Optional[List[Any]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    lens,
    layer_name: str,
    inputs: Any,
    labels: List[Any],
    *,
    pool: str = "last",
    steps: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[LinearProbe, Dict[str, Any]]:
    """
    Fit a linear probe on activations captured at `layer_name`.

    Args:
        lens: ModelLens instance
        layer_name: Layer to read activations from
        inputs: A list of examples, or a single batched tensor whose leading
                dim indexes examples (aligned with `labels`)
        labels: One label per example (any hashable type)
        pool: How to reduce a per-example activation to one vector —
              "last" (last token) or "mean" (over tokens); CNN feature maps are
              always global-average-pooled
        steps, lr, weight_decay: Optimizer settings (probes are kept small and
              regularized on purpose)
        test_frac: Fraction held out for an honest accuracy estimate
        seed: Reproducible init / split

    Returns:
        (trained LinearProbe, summary with train/test accuracy, majority-class
        baseline, and the label classes).
    """
    lens.adapter.require(AnalysisCapability.HOOKS, "train_probe")
    torch.manual_seed(seed)

    X = _gather_pooled(lens, inputs, layer_name, pool)
    classes = sorted(set(labels), key=lambda v: str(v))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([class_to_idx[v] for v in labels], dtype=torch.long)

    n, d = X.shape
    if len(y) != n:
        raise ValueError(
            f"Got {n} activation vectors but {len(y)} labels — they must align."
        )

    # Train / test split
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_test = int(round(n * test_frac)) if n > 1 else 0
    n_test = min(max(n_test, 1 if n > 1 else 0), n - 1) if n > 1 else 0
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    probe = LinearProbe(d, len(classes))
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(steps):
        logits = probe(X[train_idx])
        loss = loss_fn(logits, y[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        train_acc = _accuracy(probe(X[train_idx]), y[train_idx])
        test_acc = (
            _accuracy(probe(X[test_idx]), y[test_idx]) if n_test > 0 else None
        )

    probe.classes_ = classes
    counts = torch.bincount(y, minlength=len(classes))
    baseline = (counts.max().item() / n) if n > 0 else None

    summary = {
        "layer_name": layer_name,
        "input_dim": d,
        "num_classes": len(classes),
        "classes": classes,
        "num_examples": n,
        "num_train": int(train_idx.numel()),
        "num_test": int(test_idx.numel()),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "baseline": baseline,  # majority-class accuracy (chance level)
        "pool": pool,
    }
    return probe, summary


def probe_sweep(
    lens,
    inputs: Any,
    labels: List[Any],
    *,
    layer_names: Optional[List[str]] = None,
    pool: str = "last",
    **train_kwargs,
) -> Dict[str, Any]:
    """
    Train a probe at each layer and report where the concept is most decodable.

    Args:
        lens: ModelLens instance
        inputs, labels: As in train_probe
        layer_names: Layers to sweep. Defaults to the adapter's ordered
                     sequential layers (falls back to all named layers).
        pool: Pooling passed to each probe
        **train_kwargs: Forwarded to train_probe (steps, lr, ...)

    Returns:
        Dict with per-layer accuracies, the layer_scores map, and the best
        layer by held-out accuracy.
    """
    lens.adapter.require(AnalysisCapability.HOOKS, "probe_sweep")

    if layer_names is None:
        try:
            layer_names = lens.adapter.get_sequential_layers()
        except NotImplementedError:
            layer_names = lens.layer_names()

    per_layer: List[Dict[str, Any]] = []
    for name in layer_names:
        try:
            _, summary = train_probe(
                lens, name, inputs, labels, pool=pool, **train_kwargs
            )
        except Exception:
            # A layer whose activation can't be pooled/probed is skipped.
            continue
        per_layer.append(
            {
                "layer_name": name,
                "train_accuracy": summary["train_accuracy"],
                "test_accuracy": summary["test_accuracy"],
            }
        )

    scored = [p for p in per_layer if p["test_accuracy"] is not None]
    best = max(scored, key=lambda p: p["test_accuracy"]) if scored else None

    return {
        "per_layer": per_layer,
        "layer_scores": {
            p["layer_name"]: p["test_accuracy"] for p in per_layer
        },
        "num_layers_probed": len(per_layer),
        "best_layer": best["layer_name"] if best else None,
        "best_accuracy": best["test_accuracy"] if best else None,
    }


def apply_probe(
    lens,
    inputs: Any,
    probe: LinearProbe,
    layer_name: str,
    *,
    pool: str = "last",
) -> Dict[str, Any]:
    """
    Run inputs through a trained probe at `layer_name` and return predictions.
    """
    lens.adapter.require(AnalysisCapability.HOOKS, "apply_probe")

    X = _gather_pooled(lens, inputs, layer_name, pool)
    probe.eval()
    with torch.no_grad():
        logits = probe(X)
        probs = F.softmax(logits, dim=-1)
        pred_idx = logits.argmax(dim=-1)

    classes = getattr(probe, "classes_", None)
    if classes is not None:
        predictions = [classes[i] for i in pred_idx.tolist()]
    else:
        predictions = pred_idx.tolist()

    return {
        "predictions": predictions,
        "pred_indices": pred_idx,
        "probabilities": probs,
        "num_inputs": int(X.shape[0]),
    }


# ---- Private helpers ----


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() == 0:
        return 0.0
    return (logits.argmax(dim=-1) == y).float().mean().item()


def _gather_pooled(lens, inputs: Any, layer_name: str, pool: str) -> torch.Tensor:
    """Collect one pooled activation vector per example -> (num_examples, dim)."""
    if isinstance(inputs, (list, tuple)):
        chunks = []
        for inp in inputs:
            lens.attach_layers([layer_name])
            lens.run(inp)
            act = lens.get_layer_activation(layer_name)
            if act is None:
                raise ValueError(f"No activation captured at layer '{layer_name}'.")
            chunks.append(_pool(act, pool))
        return torch.cat(chunks, dim=0)

    lens.attach_layers([layer_name])
    lens.run(inputs)
    act = lens.get_layer_activation(layer_name)
    if act is None:
        raise ValueError(f"No activation captured at layer '{layer_name}'.")
    return _pool(act, pool)


def _pool(act: torch.Tensor, pool: str) -> torch.Tensor:
    """Reduce an activation to (batch, dim), one vector per example."""
    if act.dim() == 4:
        # CNN feature maps (batch, channels, H, W) -> global average pool
        return act.mean(dim=(2, 3)).float()
    if act.dim() == 3:
        # Sequence model (batch, seq, dim)
        if pool == "mean":
            return act.mean(dim=1).float()
        return act[:, -1, :].float()  # last token
    if act.dim() == 2:
        return act.float()
    return act.reshape(act.shape[0], -1).float()
