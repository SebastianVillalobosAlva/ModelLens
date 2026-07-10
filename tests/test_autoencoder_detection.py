import torch
import torch.nn as nn

from modellens.core.lens import ModelLens
from modellens.analysis.sparse_autoencoder import SparseAutoencoder


def test_overcomplete_sae_detected_as_autoencoder():
    sae = SparseAutoencoder(input_dim=16, expansion=4)
    lens = ModelLens(sae)

    assert lens.adapter.architecture_family == "autoencoder"
    assert "dictionary_analysis" in lens.available_analyses()


def test_compressive_autoencoder_has_no_dictionary_capability():
    # Bottleneck AE: 16 -> 4 -> 16 (in == out, but hidden < input)
    ae = nn.Sequential(nn.Linear(16, 4), nn.ReLU(), nn.Linear(4, 16))
    lens = ModelLens(ae)

    assert lens.adapter.architecture_family == "autoencoder"
    assert "dictionary_analysis" not in lens.available_analyses()


def test_mlp_classifier_is_not_an_autoencoder():
    # out (5) != in (8) -> a classifier, not an autoencoder
    mlp = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 5))
    lens = ModelLens(mlp)

    assert lens.adapter.architecture_family == "feedforward"
    assert "dictionary_analysis" not in lens.available_analyses()


def test_dictionary_features_inspection():
    torch.manual_seed(0)
    sae = SparseAutoencoder(input_dim=16, expansion=4)
    lens = ModelLens(sae)

    activations = torch.randn(8, 16)
    top_k = 5
    result = lens.dictionary_features(activations, top_k=top_k)

    assert result["num_features"] == 16 * 4
    assert result["num_inputs"] == 8
    for key in ("active_features", "dead_features", "feature_stats", "per_input"):
        assert key in result

    for entry in result["per_input"]:
        assert len(entry["top_features"]) == min(top_k, result["num_features"])
        for _, value in entry["top_features"]:
            assert value >= 0  # ReLU code

    # Decoder dictionary norms are reported per active feature.
    for feat, stat in result["feature_stats"].items():
        assert stat["activation_max"] >= 0
        assert "direction_norm" in stat


def test_feature_directions_returns_vectors():
    sae = SparseAutoencoder(input_dim=16, expansion=4)
    lens = ModelLens(sae)

    result = lens.feature_directions()
    assert result["num_features"] == 16 * 4
    assert result["input_dim"] == 16
    assert result["directions"].shape == (16 * 4, 16)
    assert result["norms"].shape == (16 * 4,)
    # Encoder read-in directions line up with the feature count.
    assert result["encoder_directions"].shape == (16 * 4, 16)

    # Normalized directions are unit-norm.
    normed = lens.feature_directions(normalize=True)
    assert torch.allclose(
        normed["directions"].norm(dim=1), torch.ones(16 * 4), atol=1e-4
    )
