import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def run_residual_analysis(
    lens, inputs, layer_names: Optional[List[str]] = None, **kwargs
) -> Dict:
    """
    Analyze how information flows through the residual stream.

    In transformers, each layer adds to a running "residual stream."
    This analysis measures how much each layer contributes to the
    final representation by comparing activations before and after each layer.

    Args:
        lens: ModelLens instance
        inputs: Model input
        layer_names: Layers to analyze. If None, uses all hooked layers.

    Returns:
        Dict with per-layer contribution metrics
    """
    # Attach hooks and run
    if layer_names:
        lens.attach_layers(layer_names)
    elif len(lens.hooks) == 0:
        lens.attach_all()

    output = lens.run(inputs, **kwargs)
    activations = lens.get_activations()

    # Filter to requested layers
    if layer_names:
        activations = {k: v for k, v in activations.items() if k in layer_names}

    # Get ordered list of layer activations
    layer_list = list(activations.items())

    if len(layer_list) < 2:
        raise ValueError("Need at least 2 layers to analyze residual stream.")

    contributions = {}
    for i in range(1, len(layer_list)):
        prev_name, prev_act = layer_list[i - 1]
        curr_name, curr_act = layer_list[i]

        # Only compare if shapes match (same dimension = same residual stream)
        if prev_act.shape != curr_act.shape:
            continue

        # What this layer added to the residual stream
        delta = curr_act - prev_act

        # Magnitude of the contribution
        delta_norm = torch.norm(delta, dim=-1).mean().item()

        # Magnitude of the running stream
        stream_norm = torch.norm(curr_act, dim=-1).mean().item()

        # Cosine similarity between consecutive layers (how much direction changed)
        cos_sim = (
            F.cosine_similarity(
                prev_act.reshape(-1, prev_act.shape[-1]),
                curr_act.reshape(-1, curr_act.shape[-1]),
                dim=-1,
            )
            .mean()
            .item()
        )

        # Relative contribution: how big is this layer's update vs the stream
        relative_contribution = delta_norm / (stream_norm + 1e-10)

        contributions[curr_name] = {
            "delta_norm": delta_norm,
            "stream_norm": stream_norm,
            "cosine_similarity": cos_sim,
            "relative_contribution": relative_contribution,
            "prev_layer": prev_name,
        }

    return {
        "contributions": contributions,
        "num_layers_analyzed": len(contributions),
        "total_stream_change": _total_change(layer_list),
    }


def _total_change(layer_list: list) -> Dict:
    """Compare the first and last layer activations for overall change."""
    first_name, first_act = layer_list[0]
    last_name, last_act = layer_list[-1]

    if first_act.shape != last_act.shape:
        return {"comparable": False}

    total_delta = torch.norm(last_act - first_act, dim=-1).mean().item()
    cos_sim = (
        F.cosine_similarity(
            first_act.reshape(-1, first_act.shape[-1]),
            last_act.reshape(-1, last_act.shape[-1]),
            dim=-1,
        )
        .mean()
        .item()
    )

    return {
        "comparable": True,
        "first_layer": first_name,
        "last_layer": last_name,
        "total_delta_norm": total_delta,
        "cosine_similarity": cos_sim,
    }


def identify_critical_layers(
    residual_results: Dict, threshold: float = 0.1
) -> List[str]:
    """
    Identify layers with the highest relative contribution to the
    residual stream. These are the layers that change the representation
    the most and are likely the most important for the model's behavior.

    Args:
        residual_results: Output from run_residual_analysis
        threshold: Minimum relative contribution to be considered "critical"

    Returns:
        List of layer names sorted by contribution (highest first)
    """
    contributions = residual_results["contributions"]

    critical = [
        (name, data["relative_contribution"])
        for name, data in contributions.items()
        if data["relative_contribution"] >= threshold
    ]

    # Sort by contribution, highest first
    critical.sort(key=lambda x: x[1], reverse=True)

    return [name for name, _ in critical]
