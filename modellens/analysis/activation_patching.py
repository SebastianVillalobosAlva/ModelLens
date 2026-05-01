import torch
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager


@contextmanager
def _hook_context():
    """
    Context manager that collects hook handles and guarantees
    they are removed on exit — even if an exception is raised.
    """
    hooks = []
    try:
        yield hooks
    finally:
        for h in hooks:
            h.remove()


def run_activation_patching(
    lens,
    clean_input,
    corrupted_input,
    layer_names: Optional[List[str]] = None,
    metric_fn: Optional[Callable] = None,
    **kwargs,
) -> Dict:
    """
    Run activation patching: replace activations from a clean run with those
    from a corrupted run to measure each sublayer's causal impact.

    Patches at the sublayer level (attn, mlp) rather than whole blocks,
    since whole-block patching corrupts the residual stream too aggressively.

    Args:
        lens: ModelLens instance
        clean_input: The input that produces the "correct" behavior
        corrupted_input: A modified input that produces different behavior
        layer_names: Which layers to patch. If None, patches all attn and mlp
                     sublayers automatically.
        metric_fn: Function(output) -> float to measure behavior change.
                   If None, uses the max logit at the last position.

    Returns:
        Dict with patching effects per layer
    """
    if metric_fn is None:
        metric_fn = _default_metric

    model = lens.model
    available = dict(model.named_modules())

    # Input length validation
    clean_len = _get_seq_length(clean_input)
    corrupted_len = _get_seq_length(corrupted_input)
    if clean_len and corrupted_len and clean_len != corrupted_len:
        raise ValueError(
            f"Clean ({clean_len}) and corrupted ({corrupted_len}) inputs "
            f"must have the same token length."
        )

    # Auto-detect sublayers if not specified
    if layer_names is None:
        layer_names = _get_sublayers(model)

    # Step 1: Get clean metric
    with torch.no_grad():
        clean_output = _forward(model, clean_input, **kwargs)
    clean_metric = metric_fn(clean_output)

    # Step 2: Capture corrupted activations (hooks scoped and cleaned up)
    corrupted_activations, corrupted_output = _capture_activations(
        model, available, corrupted_input, layer_names, **kwargs
    )
    corrupted_metric = metric_fn(corrupted_output)

    # Step 3: Patch one sublayer at a time (each patch cleans up its hook)
    patch_effects = {}
    for target_layer in layer_names:
        patched_output = _run_with_patch(
            model,
            available,
            clean_input,
            target_layer,
            corrupted_activations[target_layer],
            **kwargs,
        )
        patched_metric = metric_fn(patched_output)
        effect = patched_metric - clean_metric
        total_effect = corrupted_metric - clean_metric

        patch_effects[target_layer] = {
            "patched_metric": patched_metric,
            "effect": effect,
            "normalized_effect": effect / (total_effect + 1e-10),
        }

    return {
        "clean_metric": clean_metric,
        "corrupted_metric": corrupted_metric,
        "total_effect": corrupted_metric - clean_metric,
        "patch_effects": patch_effects,
    }


def _capture_activations(model, available, inputs, layer_names, **kwargs):
    """Capture activations at specified layers during a forward pass."""
    activations = {}

    with _hook_context() as hooks:
        for name in layer_names:

            def make_hook(n):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        activations[n] = tuple(
                            o.detach().clone() if o is not None else None
                            for o in output
                        )
                    else:
                        activations[n] = output.detach().clone()

                return hook_fn

            hook = available[name].register_forward_hook(make_hook(name))
            hooks.append(hook)

        with torch.no_grad():
            output = _forward(model, inputs, **kwargs)

    # Hooks are automatically removed here by _hook_context

    return activations, output


def _run_with_patch(model, available, inputs, target_layer, patch_activation, **kwargs):
    """Run the model with a single layer's activation replaced."""

    with _hook_context() as hooks:

        def patch_hook(module, input, output, pa=patch_activation):
            return pa

        hook = available[target_layer].register_forward_hook(patch_hook)
        hooks.append(hook)

        with torch.no_grad():
            output = _forward(model, inputs, **kwargs)

    # Hook is automatically removed here by _hook_context

    return output


def _forward(model, inputs, **kwargs):
    """Run forward pass handling different input types."""
    if isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
        return model(**inputs, **kwargs)
    return model(inputs, **kwargs)


def _get_seq_length(inputs) -> Optional[int]:
    """Get sequence length from inputs if possible."""
    if hasattr(inputs, "input_ids"):
        return inputs["input_ids"].shape[-1]
    if isinstance(inputs, dict) and "input_ids" in inputs:
        return inputs["input_ids"].shape[-1]
    if isinstance(inputs, torch.Tensor):
        return inputs.shape[-1]
    return None


def _get_sublayers(model) -> List[str]:
    """Auto-detect attn and mlp sublayers for patching."""
    sublayers = []
    for name, _ in model.named_modules():
        if name.endswith(".attn") or name.endswith(".mlp"):
            sublayers.append(name)
        elif name.endswith(".self_attn") or name.endswith(".self_attention"):
            sublayers.append(name)
    return sublayers


def _default_metric(output) -> float:
    """
    Default metric: return the max logit value at the last token position.
    Useful for language models where we care about the predicted next token.
    """
    if hasattr(output, "logits"):
        output = output.logits
    return output[:, -1, :].max(dim=-1).values.mean().item()
