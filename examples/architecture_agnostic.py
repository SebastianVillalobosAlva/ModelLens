"""
Architecture-agnostic worked example.

ModelLens's differentiator: the *same* API auto-detects each architecture and
exposes the analyses that make sense for it — no transformer assumptions. This
script points ModelLens at a CNN, an LSTM, and an MLP (all tiny, no downloads)
and runs the architecture-appropriate analysis on each.

Run:
    python examples/architecture_agnostic.py
"""

import torch
import torch.nn as nn

from modellens import ModelLens


def tiny_cnn() -> nn.Module:
    """A small conv net (image classifier shape)."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


class TinyLSTM(nn.Module):
    """A small sequence classifier: embedding -> LSTM -> linear head."""

    def __init__(self, vocab_size=50, hidden=16, num_classes=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        embedded = self.embed(x)
        out, _ = self.lstm(embedded)
        return self.head(out[:, -1, :])


def tiny_mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 3))


def describe(name: str, lens: ModelLens) -> None:
    print(f"\n=== {name} ===")
    print(f"auto-detected architecture: {lens.adapter.architecture_family}")
    print(f"available analyses: {lens.available_analyses()}")


def main() -> None:
    torch.manual_seed(0)

    # ---- CNN: filter + feature-map analysis ----
    cnn_lens = ModelLens(tiny_cnn())
    describe("CNN", cnn_lens)
    image = torch.randn(1, 3, 32, 32)

    filters = cnn_lens.filter_analysis(image)
    print(
        f"filter_analysis: {filters['total_filters']} filters, "
        f"{filters['total_dead_filters']} dead "
        f"({filters['dead_filter_ratio']:.0%})"
    )
    fmaps = cnn_lens.feature_maps(image)
    print(
        f"feature_maps: {fmaps['num_layers_tracked']} layers, "
        f"spatial_reduction={fmaps['spatial_reduction']}x, "
        f"channel_expansion={fmaps['channel_expansion']}x"
    )

    # ---- LSTM: gate analysis ----
    lstm_lens = ModelLens(TinyLSTM())
    describe("LSTM", lstm_lens)
    tokens = torch.randint(0, 50, (1, 12))

    gates = lstm_lens.gate_analysis(tokens)
    for layer_name, data in gates["layer_results"].items():
        print(
            f"gate_analysis[{layer_name}]: type={data['type']}, "
            f"hidden_size={data['hidden_size']}, "
            f"final_hidden_norm={data['final_hidden_norm']:.3f}"
        )

    # ---- MLP: layer probing (works on anything with an output projection) ----
    mlp_lens = ModelLens(tiny_mlp())
    describe("MLP", mlp_lens)
    probe = mlp_lens.layer_probe(torch.randn(1, 20), top_k=3)
    print(f"layer_probe: projected {len(probe['layer_results'])} layers")

    # ---- Same API, capability-checked: an unsupported call is a clear error ----
    print("\n=== unified capability checking ===")
    try:
        cnn_lens.gate_analysis(image)  # gates only exist on recurrent models
    except Exception as exc:
        print(f"cnn_lens.gate_analysis(...) -> {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
