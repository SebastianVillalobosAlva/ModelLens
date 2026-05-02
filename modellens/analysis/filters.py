import torch
import torch.nn as nn
from typing import Dict, List, Optional


def run_filter_analysis(
    lens, inputs, layer_names: Optional[List[str]] = None, **kwargs
) -> Dict:
    """
    Analyze CNN filters: capture feature maps during a forward pass
    and compute per-filter statistics.

    Args:
        lens: ModelLens instance with CNN adapter
        inputs: Image tensor (batch, channels, H, W)
        layer_names: Conv layers to analyze. If None, analyzes all.

    Returns:
        Dict with per-layer feature maps, activation stats, and
        dead filter detection
    """
    if layer_names is None:
        layer_names = lens.adapter.get_conv_layers()

    if not layer_names:
        raise ValueError("No convolutional layers found in the model.")

    # Attach hooks and run
    lens.attach_layers(layer_names)
    output = lens.run(inputs, **kwargs)
    activations = lens.get_activations()

    results = {}
    total_dead = 0
    total_filters = 0

    for name in layer_names:
        if name not in activations:
            continue

        feature_map = activations[name]

        # Handle different feature map dimensions
        if feature_map.dim() == 4:
            # Standard: (batch, filters, H, W)
            num_filters = feature_map.shape[1]
            spatial = (feature_map.shape[2], feature_map.shape[3])

            mean_act = feature_map.mean(dim=(0, 2, 3))
            max_act = feature_map.amax(dim=(0, 2, 3))
            std_act = feature_map.std(dim=(0, 2, 3))

            # Dead filters: max activation is 0 across entire spatial map
            dead_mask = max_act == 0
            dead_count = dead_mask.sum().item()

            # Sparsity: fraction of zero activations per filter
            sparsity = (feature_map == 0).float().mean(dim=(0, 2, 3))

        elif feature_map.dim() == 3:
            # 1D conv: (batch, filters, length)
            num_filters = feature_map.shape[1]
            spatial = (feature_map.shape[2],)

            mean_act = feature_map.mean(dim=(0, 2))
            max_act = feature_map.amax(dim=(0, 2))
            std_act = feature_map.std(dim=(0, 2))

            dead_mask = max_act == 0
            dead_count = dead_mask.sum().item()
            sparsity = (feature_map == 0).float().mean(dim=(0, 2))

        else:
            continue

        total_dead += dead_count
        total_filters += num_filters

        results[name] = {
            "feature_map": feature_map,
            "num_filters": num_filters,
            "spatial_size": spatial,
            "mean_activation": mean_act,
            "max_activation": max_act,
            "std_activation": std_act,
            "sparsity_per_filter": sparsity,
            "dead_filters": dead_count,
            "dead_filter_indices": dead_mask.nonzero(as_tuple=True)[0].tolist(),
        }

    # Get filter weight info
    filter_info = _get_filter_info(lens, layer_names)

    return {
        "layer_results": results,
        "filter_info": filter_info,
        "total_dead_filters": total_dead,
        "total_filters": total_filters,
        "dead_filter_ratio": total_dead / max(total_filters, 1),
    }


def run_feature_map_analysis(
    lens, inputs, layer_names: Optional[List[str]] = None, **kwargs
) -> Dict:
    """
    Track how feature maps evolve through the network.

    Measures spatial resolution reduction, channel expansion,
    and sparsity at each layer — useful for understanding how
    the CNN progressively abstracts from raw pixels to features.

    Args:
        lens: ModelLens instance with CNN adapter
        inputs: Image tensor (batch, channels, H, W)
        layer_names: Layers to track. If None, uses sequential layers.

    Returns:
        Dict with per-layer spatial stats and cross-layer comparisons
    """
    if layer_names is None:
        layer_names = lens.adapter.get_sequential_layers()

    if not layer_names:
        raise ValueError("No layers found for feature map analysis.")

    # Attach hooks and run
    lens.attach_layers(layer_names)
    output = lens.run(inputs, **kwargs)
    activations = lens.get_activations()

    evolution = []
    for name in layer_names:
        if name not in activations:
            continue

        fm = activations[name]

        if fm.dim() == 4:
            # (batch, channels, H, W)
            entry = {
                "layer": name,
                "channels": fm.shape[1],
                "spatial_h": fm.shape[2],
                "spatial_w": fm.shape[3],
                "total_activations": fm.shape[1] * fm.shape[2] * fm.shape[3],
                "sparsity": (fm == 0).float().mean().item(),
                "mean_activation": fm.mean().item(),
                "std_activation": fm.std().item(),
            }
        elif fm.dim() == 3:
            # (batch, channels, length) — 1D conv or recurrent
            entry = {
                "layer": name,
                "channels": fm.shape[1],
                "spatial_h": fm.shape[2],
                "spatial_w": 1,
                "total_activations": fm.shape[1] * fm.shape[2],
                "sparsity": (fm == 0).float().mean().item(),
                "mean_activation": fm.mean().item(),
                "std_activation": fm.std().item(),
            }
        elif fm.dim() == 2:
            # (batch, features) — linear layer
            entry = {
                "layer": name,
                "channels": fm.shape[1],
                "spatial_h": 1,
                "spatial_w": 1,
                "total_activations": fm.shape[1],
                "sparsity": (fm == 0).float().mean().item(),
                "mean_activation": fm.mean().item(),
                "std_activation": fm.std().item(),
            }
        else:
            continue

        evolution.append(entry)

    if not evolution:
        return {"evolution": [], "spatial_reduction": None, "channel_expansion": None}

    # Compute cross-layer statistics
    first = evolution[0]
    last = evolution[-1]

    spatial_reduction = None
    if first["spatial_h"] > 0 and last["spatial_h"] > 0:
        spatial_reduction = first["spatial_h"] / last["spatial_h"]

    channel_expansion = None
    if first["channels"] > 0 and last["channels"] > 0:
        channel_expansion = last["channels"] / first["channels"]

    # Find the layer where sparsity is highest (most ReLU killing)
    sparsest = max(evolution, key=lambda e: e["sparsity"])

    return {
        "evolution": evolution,
        "num_layers_tracked": len(evolution),
        "spatial_reduction": spatial_reduction,
        "channel_expansion": channel_expansion,
        "input_spatial": (first["spatial_h"], first["spatial_w"]),
        "output_spatial": (last["spatial_h"], last["spatial_w"]),
        "sparsest_layer": sparsest["layer"],
        "max_sparsity": sparsest["sparsity"],
    }


def get_filter_weights(lens, layer_name: str) -> Dict:
    """
    Get the raw filter/kernel weights for a conv layer.

    Useful for visualizing what patterns each filter detects.
    First-layer filters on RGB images are directly interpretable
    as edge/color detectors.

    Args:
        lens: ModelLens instance
        layer_name: Name of the conv layer

    Returns:
        Dict with filter weights, shapes, and statistics
    """
    available = dict(lens.model.named_modules())
    if layer_name not in available:
        raise ValueError(f"Layer '{layer_name}' not found.")

    module = available[layer_name]
    if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise ValueError(f"Layer '{layer_name}' is not a convolutional layer.")

    weights = module.weight.detach()

    return {
        "weights": weights,
        "shape": weights.shape,
        "out_channels": weights.shape[0],
        "in_channels": weights.shape[1],
        "kernel_size": weights.shape[2:],
        "weight_mean": weights.mean().item(),
        "weight_std": weights.std().item(),
        "weight_norm_per_filter": torch.norm(weights.view(weights.shape[0], -1), dim=1),
    }


def find_most_active_filters(
    filter_results: Dict, layer_name: str, top_k: int = 5
) -> Dict:
    """
    Find the filters with the highest mean activation in a given layer.

    These are the filters that respond most strongly to the input,
    indicating which features are most prominent.

    Args:
        filter_results: Output from run_filter_analysis
        layer_name: Which layer to inspect
        top_k: Number of top filters to return

    Returns:
        Dict with top filter indices and their activation values
    """
    if layer_name not in filter_results["layer_results"]:
        raise ValueError(f"Layer '{layer_name}' not found in results.")

    data = filter_results["layer_results"][layer_name]
    mean_act = data["mean_activation"]

    top_values, top_indices = torch.topk(mean_act, k=min(top_k, len(mean_act)))

    return {
        "layer": layer_name,
        "top_filter_indices": top_indices.tolist(),
        "top_filter_activations": top_values.tolist(),
        "num_filters": data["num_filters"],
    }


def _get_filter_info(lens, layer_names: List[str]) -> List[Dict]:
    """Get metadata about conv layer filters."""
    info = []
    available = dict(lens.model.named_modules())

    for name in layer_names:
        if name not in available:
            continue
        module = available[name]
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            info.append(
                {
                    "name": name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding,
                    "num_parameters": module.weight.numel(),
                }
            )

    return info
