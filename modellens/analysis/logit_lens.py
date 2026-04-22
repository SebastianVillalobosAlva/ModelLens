import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def run_logit_lens(
    lens, inputs, layer_names: Optional[List[str]] = None, top_k: int = 5, **kwargs
) -> Dict:
    """
    Run logit lens analysis: project each layer's hidden state through the
    unembedding matrix to see what the model would predict at that layer.

    Args:
        lens: ModelLens instance with hooks attached
        inputs: Model input (string or tensor)
        layer_names: Layers to analyze. If None, uses all hooked layers.
        top_k: Number of top predictions to return per layer

    Returns:
        Dict with layer-by-layer predictions and probabilities
    """
    # Get the unembedding matrix from the adapter
    unembed = lens.adapter.get_unembedding(lens.model)
    if unembed is None:
        raise ValueError(
            "Could not find unembedding matrix. " "Model may not support logit lens."
        )

    # Attach hooks to requested layers (or all if none specified)
    if layer_names:
        lens.attach_layers(layer_names)
    elif len(lens.hooks) == 0:
        lens.attach_all()

    # Run forward pass to capture activations
    output = lens.run(inputs, **kwargs)
    activations = lens.get_activations()

    # Project each layer's activations through the unembedding matrix
    hidden_dim = unembed.shape[-1]
    results = {}
    for name, activation in activations.items():
        if layer_names and name not in layer_names:
            continue
        if activation.shape[-1] != hidden_dim:
            continue

        # activation shape: (batch, seq_len, hidden_dim)
        # unembed shape: (vocab_size, hidden_dim)
        # Project hidden states to vocabulary space
        logits = activation @ unembed.T

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Get top-k predictions for the last token position
        top_probs, top_indices = torch.topk(probs[:, -1, :], k=top_k, dim=-1)

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


def decode_logit_lens(results: Dict, tokenizer=None, vocab=None) -> Dict:
    """
    Convert logit lens token indices to readable strings.

    Args:
        results: Output from run_logit_lens
        tokenizer: HuggingFace tokenizer for decoding (for HF models)
        vocab: Dict mapping index -> label (for vanilla PyTorch models)
              e.g. {0: "0", 1: "1", ..., 112: "112"} for modular arithmetic

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
            # pass

    return decoded
