import pytest
import torch
import torch.nn as nn
from modellens import ModelLens
from modellens.analysis.filters import (
    run_filter_analysis,
    run_feature_map_analysis,
    get_filter_weights,
)


@pytest.fixture(scope="session")
def cnn_model():
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


@pytest.fixture(scope="session")
def cnn_lens(cnn_model):
    return ModelLens(cnn_model)


@pytest.fixture(scope="session")
def sample_image():
    return torch.randn(1, 1, 28, 28)


class TestFilterAnalysis:

    def test_correct_filter_counts(self, cnn_lens, sample_image):
        results = run_filter_analysis(cnn_lens, sample_image)
        assert results["layer_results"]["conv1"]["num_filters"] == 16
        assert results["layer_results"]["conv2"]["num_filters"] == 32
        assert results["total_filters"] == 48

    def test_activation_stats_shapes(self, cnn_lens, sample_image):
        results = run_filter_analysis(cnn_lens, sample_image)
        conv1 = results["layer_results"]["conv1"]
        assert conv1["mean_activation"].shape == (16,)
        assert conv1["max_activation"].shape == (16,)

    def test_dead_filter_detection(self):
        class DeadFilterCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
                self.relu1 = nn.ReLU()
                self.fc = nn.Linear(4 * 28 * 28, 2)

            def forward(self, x):
                x = self.relu1(self.conv1(x))
                return self.fc(x.flatten(1))

        model = DeadFilterCNN()
        with torch.no_grad():
            model.conv1.weight[0] = 0
            model.conv1.bias[0] = -100

        lens = ModelLens(model)
        # Hook relu1 to capture post-activation feature maps
        results = run_filter_analysis(
            lens, torch.randn(1, 1, 28, 28), layer_names=["relu1"]
        )
        assert results["total_dead_filters"] >= 1


class TestFeatureMapAnalysis:

    def test_spatial_reduction(self, cnn_lens, sample_image):
        results = run_feature_map_analysis(cnn_lens, sample_image)
        assert results["num_layers_tracked"] > 0
        assert results["spatial_reduction"] > 1

    def test_sparsity_in_range(self, cnn_lens, sample_image):
        results = run_feature_map_analysis(cnn_lens, sample_image)
        for entry in results["evolution"]:
            assert 0.0 <= entry["sparsity"] <= 1.0


class TestFilterWeights:

    def test_correct_shape(self, cnn_lens):
        result = get_filter_weights(cnn_lens, "conv1")
        assert result["shape"] == (16, 1, 3, 3)
        assert result["weight_norm_per_filter"].shape == (16,)

    def test_non_conv_layer_raises(self, cnn_lens):
        with pytest.raises(ValueError, match="not a convolutional"):
            get_filter_weights(cnn_lens, "classifier")
