"""Layer-selection utilities shared across analyses."""

from typing import Optional

from modellens.adapters.base import AnalysisCapability


def default_residual_layer(lens) -> Optional[str]:
    """
    Pick a residual-stream hook site, reusing the same layer lookup that the
    residual_stream analysis uses (the adapter's ordered sequential layers).
    Returns None when the model has no residual stream, so the caller can
    require an explicit layer_name instead.
    """
    if not lens.adapter.supports(AnalysisCapability.RESIDUAL_STREAM):
        return None
    try:
        seq_layers = lens.adapter.get_sequential_layers()
    except NotImplementedError:
        return None
    if not seq_layers:
        return None
    # Middle of the stack — a feature-rich point in the residual stream.
    return seq_layers[len(seq_layers) // 2]
