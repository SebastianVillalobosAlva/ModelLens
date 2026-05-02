import torch
import torch.nn as nn
from typing import Dict, List, Optional
from contextlib import contextmanager


@contextmanager
def _hook_context():
    """Context manager that collects hooks and removes them on exit."""
    hooks = []
    try:
        yield hooks
    finally:
        for h in hooks:
            h.remove()


def run_gate_analysis(
    lens, inputs, layer_names: Optional[List[str]] = None, **kwargs
) -> Dict:
    """
    Analyze gate activations in recurrent models (LSTM, GRU).

    Captures the hidden states at each timestep and computes
    gate activation statistics to understand how the model
    processes sequential information.

    Args:
        lens: ModelLens instance
        inputs: Input tensor (batch, seq_len) for token ids, or
                (batch, seq_len, features) for continuous inputs
        layer_names: Recurrent layers to analyze. If None, auto-detects.

    Returns:
        Dict with per-layer gate statistics and hidden state evolution
    """
    if layer_names is None:
        layer_names = lens.adapter.get_gate_layers()

    if not layer_names:
        raise ValueError("No recurrent layers found in the model.")

    model = lens.model
    available = dict(model.named_modules())

    results = {}
    for name in layer_names:
        if name not in available:
            continue

        module = available[name]

        if isinstance(module, nn.LSTM):
            results[name] = _analyze_lstm(lens, module, name, inputs, **kwargs)
        elif isinstance(module, nn.GRU):
            results[name] = _analyze_gru(lens, module, name, inputs, **kwargs)

    return {
        "layer_results": results,
        "num_layers_analyzed": len(results),
    }


def _analyze_lstm(lens, module, layer_name, inputs, **kwargs) -> Dict:
    """
    Analyze LSTM gate activations.

    PyTorch's LSTM computes all gates internally, so we capture
    the outputs and decompose the weight matrices to infer gate behavior.
    """
    captured = {}

    with _hook_context() as hooks:

        def capture_hook(mod, inp, out):
            output, (h_n, c_n) = out
            captured["output"] = output.detach()  # (batch, seq_len, hidden*directions)
            captured["h_n"] = h_n.detach()  # (num_layers*directions, batch, hidden)
            captured["c_n"] = c_n.detach()  # (num_layers*directions, batch, hidden)
            # Capture input to LSTM for gate decomposition
            if isinstance(inp, tuple) and len(inp) > 0:
                captured["input"] = inp[0].detach()

        hook = module.register_forward_hook(capture_hook)
        hooks.append(hook)

        with torch.no_grad():
            lens.adapter.forward(inputs, **kwargs)

    if "output" not in captured:
        raise ValueError(f"Failed to capture outputs from LSTM layer '{layer_name}'.")

    output = captured["output"]
    h_n = captured["h_n"]
    c_n = captured["c_n"]

    hidden_size = module.hidden_size
    num_layers = module.num_layers
    bidirectional = module.bidirectional
    directions = 2 if bidirectional else 1

    # Decompose gate weights for each LSTM layer
    gate_stats = _decompose_lstm_gates(module, num_layers, directions, hidden_size)

    # Analyze hidden state evolution over sequence
    # output shape: (batch, seq_len, hidden_size * directions)
    hidden_evolution = _analyze_hidden_evolution(output)

    # Cell state analysis
    cell_state = {
        "final_cell_norm": torch.norm(c_n, dim=-1).mean().item(),
        "final_cell_mean": c_n.mean().item(),
        "final_cell_std": c_n.std().item(),
    }

    return {
        "type": "lstm",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "gate_weight_stats": gate_stats,
        "hidden_evolution": hidden_evolution,
        "cell_state": cell_state,
        "final_hidden_norm": torch.norm(h_n, dim=-1).mean().item(),
        "output_shape": list(output.shape),
    }


def _analyze_gru(lens, module, layer_name, inputs, **kwargs) -> Dict:
    """Analyze GRU gate activations."""
    captured = {}

    with _hook_context() as hooks:

        def capture_hook(mod, inp, out):
            output, h_n = out
            captured["output"] = output.detach()
            captured["h_n"] = h_n.detach()

        hook = module.register_forward_hook(capture_hook)
        hooks.append(hook)

        with torch.no_grad():
            lens.adapter.forward(inputs, **kwargs)

    if "output" not in captured:
        raise ValueError(f"Failed to capture outputs from GRU layer '{layer_name}'.")

    output = captured["output"]
    h_n = captured["h_n"]

    hidden_size = module.hidden_size
    num_layers = module.num_layers
    bidirectional = module.bidirectional
    directions = 2 if bidirectional else 1

    gate_stats = _decompose_gru_gates(module, num_layers, directions, hidden_size)
    hidden_evolution = _analyze_hidden_evolution(output)

    return {
        "type": "gru",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "gate_weight_stats": gate_stats,
        "hidden_evolution": hidden_evolution,
        "final_hidden_norm": torch.norm(h_n, dim=-1).mean().item(),
        "output_shape": list(output.shape),
    }


def _decompose_lstm_gates(module, num_layers, directions, hidden_size) -> List[Dict]:
    """
    Decompose LSTM weight matrices into individual gate statistics.

    PyTorch LSTM packs 4 gates into weight_ih and weight_hh:
    [input_gate, forget_gate, cell_gate, output_gate]
    Each slice has size hidden_size.
    """
    gate_names = ["input", "forget", "cell", "output"]
    all_stats = []

    for layer_idx in range(num_layers):
        for direction in range(directions):
            suffix = f"_reverse" if direction == 1 else ""
            prefix = f"l{layer_idx}{suffix}"

            # Access weight matrices
            w_ih = getattr(module, f"weight_ih_{prefix}", None)
            w_hh = getattr(module, f"weight_hh_{prefix}", None)

            if w_ih is None or w_hh is None:
                continue

            layer_gates = {"layer": layer_idx, "direction": direction}

            for gate_idx, gate_name in enumerate(gate_names):
                start = gate_idx * hidden_size
                end = (gate_idx + 1) * hidden_size

                ih_slice = w_ih[start:end].detach()
                hh_slice = w_hh[start:end].detach()

                layer_gates[gate_name] = {
                    "input_weight_norm": torch.norm(ih_slice).item(),
                    "hidden_weight_norm": torch.norm(hh_slice).item(),
                    "input_weight_mean": ih_slice.mean().item(),
                    "hidden_weight_mean": hh_slice.mean().item(),
                }

            all_stats.append(layer_gates)

    return all_stats


def _decompose_gru_gates(module, num_layers, directions, hidden_size) -> List[Dict]:
    """
    Decompose GRU weight matrices into individual gate statistics.

    PyTorch GRU packs 3 gates into weight_ih and weight_hh:
    [reset_gate, update_gate, new_gate]
    """
    gate_names = ["reset", "update", "new"]
    all_stats = []

    for layer_idx in range(num_layers):
        for direction in range(directions):
            suffix = f"_reverse" if direction == 1 else ""
            prefix = f"l{layer_idx}{suffix}"

            w_ih = getattr(module, f"weight_ih_{prefix}", None)
            w_hh = getattr(module, f"weight_hh_{prefix}", None)

            if w_ih is None or w_hh is None:
                continue

            layer_gates = {"layer": layer_idx, "direction": direction}

            for gate_idx, gate_name in enumerate(gate_names):
                start = gate_idx * hidden_size
                end = (gate_idx + 1) * hidden_size

                ih_slice = w_ih[start:end].detach()
                hh_slice = w_hh[start:end].detach()

                layer_gates[gate_name] = {
                    "input_weight_norm": torch.norm(ih_slice).item(),
                    "hidden_weight_norm": torch.norm(hh_slice).item(),
                    "input_weight_mean": ih_slice.mean().item(),
                    "hidden_weight_mean": hh_slice.mean().item(),
                }

            all_stats.append(layer_gates)

    return all_stats


def _analyze_hidden_evolution(output: torch.Tensor) -> Dict:
    """
    Analyze how hidden states change across the sequence.

    Args:
        output: LSTM/GRU output (batch, seq_len, hidden_size)

    Returns:
        Dict with per-timestep statistics
    """
    # Norm at each timestep
    norms = torch.norm(output, dim=-1)  # (batch, seq_len)
    mean_norms = norms.mean(dim=0)  # (seq_len,)

    # How much the hidden state changes between timesteps
    if output.shape[1] > 1:
        deltas = output[:, 1:, :] - output[:, :-1, :]
        delta_norms = torch.norm(deltas, dim=-1).mean(dim=0)  # (seq_len-1,)
    else:
        delta_norms = torch.tensor([0.0])

    return {
        "timestep_norms": mean_norms.tolist(),
        "timestep_deltas": delta_norms.tolist(),
        "norm_trend": "increasing" if mean_norms[-1] > mean_norms[0] else "decreasing",
        "max_delta_timestep": (
            delta_norms.argmax().item() + 1 if len(delta_norms) > 0 else 0
        ),
        "mean_hidden_norm": mean_norms.mean().item(),
    }
