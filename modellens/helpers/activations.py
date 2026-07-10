"""Activation-shape utilities shared across analyses."""

from typing import Any, Dict, List, Optional, Tuple

import torch


def normalize_activation(activation: torch.Tensor, hidden_dim: int):
    """
    Normalize an activation tensor to be compatible with the output projection.

    Handles different shapes from different architectures:
    - Transformers: (batch, seq_len, hidden_dim)
    - MLPs: (batch, hidden_dim)
    - CNNs: (batch, channels, H, W) -> global average pool to (batch, channels)

    Returns None if the activation cannot be matched to hidden_dim.
    """
    if activation.shape[-1] == hidden_dim:
        return activation

    # CNN feature maps: pool spatial dimensions
    if activation.dim() == 4 and activation.shape[1] == hidden_dim:
        return activation.mean(dim=(2, 3))  # Global average pool

    return None


def gather_activation_rows(
    lens, inp: Any, layer_name: str, tokenizer=None, **kwargs
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Run one forward pass through `lens` and return the activation captured at
    `layer_name` flattened to (num_rows, feature_dim) rows, with provenance
    (the token position each row came from).
    """
    lens.attach_layers([layer_name])
    lens.run(inp, **kwargs)
    act = lens.get_layer_activation(layer_name)
    if act is None:
        raise ValueError(f"No activation captured at layer '{layer_name}'.")
    return flatten_activation(act, inp, tokenizer)


def flatten_activation(
    act: torch.Tensor, inp: Any = None, tokenizer=None
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Flatten an activation tensor to (num_rows, feature_dim) and record the
    token position each row came from (with a decoded token string when a
    tokenizer and a matching string input are available).
    """
    d = act.shape[-1]

    if act.dim() == 3:
        # (batch, seq, d) — sequence model
        _, seq_len, _ = act.shape
        rows = act.reshape(-1, d)
        tokens = token_strings(inp, tokenizer, seq_len)
        prov = []
        for r in range(rows.shape[0]):
            pos = r % seq_len
            prov.append({"position": pos, "token": tokens[pos] if tokens else None})
    elif act.dim() == 2:
        # (batch, d) — non-sequence model
        rows = act
        prov = [{"position": 0, "token": None} for _ in range(rows.shape[0])]
    else:
        rows = act.reshape(-1, d)
        prov = [{"position": r, "token": None} for r in range(rows.shape[0])]

    return rows.float(), prov


def token_strings(inp: Any, tokenizer, seq_len: int) -> Optional[List[str]]:
    """Best-effort decode of a string input into per-position token strings."""
    if tokenizer is None or not isinstance(inp, str):
        return None
    try:
        ids = tokenizer(inp)["input_ids"]
    except Exception:
        return None
    if len(ids) != seq_len:
        return None
    try:
        return [tokenizer.decode([i]) for i in ids]
    except Exception:
        return None
