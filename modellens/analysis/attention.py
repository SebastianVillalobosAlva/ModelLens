import torch
from typing import Dict, List, Optional


def run_attention_analysis(
    lens, inputs, layer_names: Optional[List[str]] = None, **kwargs
) -> Dict:
    """
    Extract attention weight maps from the model.

    Args:
        lens: ModelLens instance
        inputs: Model input (string or tensor)
        layer_names: Specific attention layers to analyze. If None, auto-detects.

    Returns:
        Dict with attention weights per layer
    """
    # Find attention layers if not specified
    if layer_names is None:
        layer_names = lens.adapter.get_attention_layers(lens.model)

    if not layer_names:
        raise ValueError("No attention layers found in the model.")

    # For HuggingFace models, we can use output_attentions=True
    if lens.adapter.type_of_adapter == "huggingface":
        return _extract_hf_attention(lens, inputs, layer_names, **kwargs)

    # For vanilla PyTorch, use hooks to capture attention weights
    return _extract_hook_attention(lens, inputs, layer_names, **kwargs)


def _extract_hf_attention(lens, inputs, layer_names, **kwargs) -> Dict:
    """Extract attention using HuggingFace's built-in output_attentions flag."""
    # Tokenize if needed
    if isinstance(inputs, str):
        tokens = lens.adapter.tokenize(inputs)
    else:
        tokens = inputs if isinstance(inputs, dict) else {"input_ids": inputs}

    # Run with attention output enabled
    with torch.no_grad():
        output = lens.model(**tokens, output_attentions=True, **kwargs)

    attentions = output.attentions  # Tuple of (batch, heads, seq, seq) per layer
    results = {}
    for i, attn in enumerate(attentions):
        # Match to layer name if possible, otherwise use index
        name = layer_names[i] if i < len(layer_names) else f"layer_{i}"
        results[name] = {
            "weights": attn.detach(),  # (batch, heads, seq, seq)
            "num_heads": attn.shape[1],
            "seq_length": attn.shape[-1],
        }

    return {
        "attention_maps": results,
        "num_layers": len(attentions),
    }


def _extract_hook_attention(lens, inputs, layer_names, **kwargs) -> Dict:
    """Extract attention weights using hooks for vanilla PyTorch models."""
    attention_weights = {}

    def make_attn_hook(name):
        def hook_fn(module, input, output):
            # Attention modules typically return (output, weights) or just weights
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights[name] = output[1].detach()
            else:
                attention_weights[name] = output.detach()

        return hook_fn

    # Attach custom hooks to attention layers
    for name in layer_names:
        lens.hooks.attach_custom(lens.model, name, make_attn_hook(name))

    # Run forward pass
    with torch.no_grad():
        if isinstance(inputs, dict):
            output = lens.model(**inputs, **kwargs)
        else:
            output = lens.model(inputs, **kwargs)

    results = {}
    for name, weights in attention_weights.items():
        results[name] = {
            "weights": weights,
            "num_heads": weights.shape[1] if weights.dim() >= 3 else 1,
            "seq_length": weights.shape[-1],
        }

    return {
        "attention_maps": results,
        "num_layers": len(results),
    }


def head_summary(attention_results: Dict) -> Dict:
    """
    Compute summary statistics for each attention head.
    Useful for identifying which heads are most "focused" vs "diffuse".

    Returns:
        Dict with entropy and max attention per head per layer
    """
    summary = {}
    for name, data in attention_results["attention_maps"].items():
        weights = data["weights"]

        if weights.dim() == 4:
            # HuggingFace: (batch, heads, seq, seq)
            entropy = (
                -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean(dim=(0, 2))
            )
            max_attn = weights.max(dim=-1).values.mean(dim=(0, 2))
        else:
            # Vanilla PyTorch: (batch, seq, seq) — no head dimension
            entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean(dim=0)
            max_attn = weights.max(dim=-1).values.mean(dim=0)

        summary[name] = {
            "entropy": entropy.tolist(),
            "max_attention": max_attn.tolist(),
        }

    return summary
