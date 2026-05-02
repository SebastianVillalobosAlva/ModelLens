import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def run_logit_lens(
    lens, inputs, layer_names: Optional[List[str]] = None, top_k: int = 5, **kwargs
) -> Dict:
    """
    Run layer probing (logit lens) analysis: project each layer's hidden
    state through the output projection to see what the model would
    predict at that layer.

    Works on any architecture with a hidden → output mapping
    (transformers, RNNs, MLPs, CNNs with classifiers).

    Args:
        lens: ModelLens instance
        inputs: Model input (string, tensor, or dict)
        layer_names: Layers to analyze. If None, uses all hooked layers.
        top_k: Number of top predictions to return per layer

    Returns:
        Dict with layer-by-layer predictions and probabilities
    """
    # Get the output projection matrix from the adapter
    output_proj = lens.adapter.get_output_projection()
    if output_proj is None:
        raise ValueError(
            "Could not find output projection matrix. "
            "Model may not support layer probing."
        )

    # Attach hooks to requested layers (or all if none specified)
    if layer_names:
        lens.attach_layers(layer_names)
    elif len(lens.hooks) == 0:
        lens.attach_all()

    # Run forward pass to capture activations
    output = lens.run(inputs, **kwargs)
    activations = lens.get_activations()

    # Project each layer's activations through the output projection
    hidden_dim = output_proj.shape[-1]
    results = {}
    for name, activation in activations.items():
        if layer_names and name not in layer_names:
            continue

        # Handle different activation shapes
        act = _normalize_activation(activation, hidden_dim)
        if act is None:
            continue

        # activation shape: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        # output_proj shape: (output_size, hidden_dim)
        logits = act @ output_proj.T

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Get top-k predictions at the last position
        if probs.dim() == 3:
            target_probs = probs[:, -1, :]
        else:
            target_probs = probs

        top_probs, top_indices = torch.topk(target_probs, k=top_k, dim=-1)

        results[name] = {
            "logits": logits,
            "probs": probs,
            "top_k_indices": top_indices,
            "top_k_probs": top_probs,
        }

    return {
        "layer_results": results,
        "final_output": output,
    }


def _normalize_activation(activation: torch.Tensor, hidden_dim: int):
    """
    Normalize activation tensor to be compatible with the output projection.

    Handles different shapes from different architectures:
    - Transformers: (batch, seq_len, hidden_dim)
    - MLPs: (batch, hidden_dim)
    - CNNs: (batch, channels, H, W) → global average pool to (batch, channels)
    """
    if activation.shape[-1] == hidden_dim:
        return activation

    # CNN feature maps: pool spatial dimensions
    if activation.dim() == 4 and activation.shape[1] == hidden_dim:
        return activation.mean(dim=(2, 3))  # Global average pool

    return None


def decode_logit_lens(results: Dict, tokenizer=None, vocab=None) -> Dict:
    """
    Convert layer probing token indices to readable strings.

    Args:
        results: Output from run_logit_lens
        tokenizer: HuggingFace tokenizer for decoding (for HF models)
        vocab: Dict mapping index -> label (for vanilla PyTorch models)
              e.g. {0: "cat", 1: "dog", ..., 9: "truck"} for CIFAR-10

    Returns:
        Dict mapping layer names to lists of (token, probability) pairs
    """
    if tokenizer is None and vocab is None:
        raise ValueError("Provide either a tokenizer or a vocab dict.")

    decoded = {}
    for name, data in results["layer_results"].items():
        indices = data["top_k_indices"][0]  # First batch element
        probs = data["top_k_probs"][0]

        if tokenizer:
            decoded[name] = [
                (tokenizer.decode(idx.item()), prob.item())
                for idx, prob in zip(indices, probs)
            ]
        else:
            decoded[name] = [
                (vocab.get(idx.item(), f"[{idx.item()}]"), prob.item())
                for idx, prob in zip(indices, probs)
            ]

    return decoded
