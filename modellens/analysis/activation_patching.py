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

    # Input length validation (sequence models only)
    clean_len = _get_seq_length(clean_input)
    corrupted_len = _get_seq_length(corrupted_input)
    if clean_len and corrupted_len and clean_len != corrupted_len:
        raise ValueError(
            f"Clean ({clean_len}) and corrupted ({corrupted_len}) inputs "
            f"must have the same token length."
        )

    # Auto-detect sublayers if not specified
    if layer_names is None:
        layer_names = lens.adapter.get_patchable_layers()
    # Guaranteed to be a list at this point
    assert layer_names is not None, "No patchable layers found."

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


def run_attribution_patching(
    lens,
    clean_input,
    corrupted_input,
    layer_names: Optional[List[str]] = None,
    metric_fn: Optional[Callable] = None,
    **kwargs,
) -> Dict:
    """
    Attribution patching: a first-order (gradient) approximation of activation
    patching.

    Instead of one forward pass per layer, it estimates every sublayer's effect
    from ~2 forward passes + 1 backward pass, using the linear approximation

        effect ≈ (a_corrupt - a_clean) · ∂metric/∂a_clean

    evaluated at the clean activation. This is *exact* when the network and
    metric are linear, and a Taylor approximation otherwise (Nanda 2023; Syed
    et al. 2023, "Attribution Patching Outperforms Automated Circuit Discovery").

    Returns the same schema as run_activation_patching (clean_metric,
    corrupted_metric, total_effect, patch_effects with effect / normalized_effect
    / patched_metric), so it is a drop-in backend for circuit discovery.

    Note: unlike run_activation_patching, `metric_fn` here must return a scalar
    *tensor* (differentiable), not a float. The default handles this.

    Args:
        lens: ModelLens instance
        clean_input: Input producing the "correct" behavior
        corrupted_input: Modified input producing different behavior
        layer_names: Sublayers to attribute. If None, uses get_patchable_layers().
        metric_fn: Differentiable metric(output) -> scalar tensor.

    Returns:
        Dict of attribution effects per layer.
    """
    if metric_fn is None:
        metric_fn = _default_metric_tensor

    model = lens.model
    available = dict(model.named_modules())

    clean_len = _get_seq_length(clean_input)
    corrupted_len = _get_seq_length(corrupted_input)
    if clean_len and corrupted_len and clean_len != corrupted_len:
        raise ValueError(
            f"Clean ({clean_len}) and corrupted ({corrupted_len}) inputs "
            f"must have the same token length."
        )

    if layer_names is None:
        layer_names = lens.adapter.get_patchable_layers()
    assert layer_names is not None, "No patchable layers found."

    # Corrupted activations (no grad, detached) — reuse the existing capture.
    corrupted_activations, corrupted_output = _capture_activations(
        model, available, corrupted_input, layer_names, **kwargs
    )
    corrupted_metric = float(metric_fn(corrupted_output))

    # Clean forward WITH grad; retain grad on each captured activation.
    clean_activations: Dict[str, torch.Tensor] = {}
    with _hook_context() as hooks:
        for name in layer_names:
            hook = available[name].register_forward_hook(
                _make_grad_hook(name, clean_activations)
            )
            hooks.append(hook)
        model.zero_grad(set_to_none=True)
        clean_output = _forward(model, clean_input, **kwargs)

    metric = metric_fn(clean_output)
    if not isinstance(metric, torch.Tensor) or metric.dim() != 0:
        raise ValueError(
            "attribution patching metric_fn must return a scalar tensor."
        )
    if not metric.requires_grad:
        raise ValueError(
            "attribution patching needs a grad-enabled model; the metric does "
            "not require grad (are the model's parameters frozen?)."
        )
    metric.backward()
    clean_metric = float(metric.detach())

    total_effect = corrupted_metric - clean_metric
    patch_effects = {}
    for name in layer_names:
        a_clean = clean_activations.get(name)
        a_corrupt = corrupted_activations.get(name)
        if a_clean is None or a_corrupt is None or a_clean.grad is None:
            attribution = 0.0
        else:
            attribution = float(
                ((a_corrupt - a_clean.detach()) * a_clean.grad).sum()
            )
        patch_effects[name] = {
            "attribution": attribution,
            "effect": attribution,
            "patched_metric": clean_metric + attribution,
            "normalized_effect": attribution / (total_effect + 1e-10),
        }

    model.zero_grad(set_to_none=True)

    return {
        "clean_metric": clean_metric,
        "corrupted_metric": corrupted_metric,
        "total_effect": total_effect,
        "patch_effects": patch_effects,
        "method": "attribution",
    }


def _make_grad_hook(name, store):
    """Forward hook that keeps the module's output tensor and retains its grad."""

    def hook_fn(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.requires_grad:
            t.retain_grad()
            store[name] = t

    return hook_fn


def _capture_activations(model, available, inputs, layer_names, **kwargs):
    """Capture activations at specified layers during a forward pass."""
    activations = {}

    with _hook_context() as hooks:
        for name in layer_names:

            def make_hook(n):
                # in _capture_activations
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        activations[n] = output[0].detach().clone()  # hidden state only
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

        # in _run_with_patch
        def patch_hook(module, input, output, pa=patch_activation):
            if isinstance(output, tuple):
                return (pa,) + tuple(output[1:])  # swap hidden, keep live weights/cache
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


def _default_metric_tensor(output) -> torch.Tensor:
    """
    Differentiable default metric: the max logit at the last token position,
    returned as a scalar tensor so it can be back-propagated (attribution
    patching).
    """
    if hasattr(output, "logits"):
        output = output.logits
    return output[:, -1, :].max(dim=-1).values.mean()


def _default_metric(output) -> float:
    """
    Default metric: return the max logit value at the last token position.
    Useful for language models where we care about the predicted next token.
    """
    return _default_metric_tensor(output).item()
