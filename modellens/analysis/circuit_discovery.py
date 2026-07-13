import re
import torch
from typing import Any, Callable, Dict, List, Optional
from modellens.adapters.base import AnalysisCapability
from modellens.analysis.activation_patching import (
    run_activation_patching,
    run_attribution_patching,
)


def discover_circuit(
    lens,
    clean_input: Any,
    corrupted_input: Any,
    *,
    metric_fn: Optional[Callable] = None,
    importance_threshold: float = 0.15,
    layer_names: Optional[List[str]] = None,
    method: str = "exact",
    **kwargs,
) -> Dict[str, Any]:
    """
    Automatically discover the causal circuit for a specific behavior.

    Step 1: Activation patching identifies which components are causally
            important (works on any architecture).
    Step 2: Attention analysis traces how information flows between
            important components (transformers only).
    Step 3: Nodes are assigned functional roles (critical, booster,
            gate, processor) based on their causal effect.

    Args:
        lens: ModelLens instance
        clean_input: Input that produces the "correct" behavior
        corrupted_input: Modified input that produces different behavior
        metric_fn: Custom metric function for patching. With method="exact" it
                   returns a float; with method="attribution" a scalar tensor.
        importance_threshold: Minimum |normalized_effect| to be included
        layer_names: Sublayers to patch. If None, uses adapter's
                     get_patchable_layers().
        method: "exact" (one forward pass per layer) or "attribution"
                (gradient approximation, ~2 passes — scales to large models).

    Returns:
        Dict with nodes, edges, and metadata for circuit visualization.
    """
    lens.adapter.require(AnalysisCapability.ACTIVATION_PATCHING, "discover_circuit")

    # Step 1: Find causally important components via activation patching
    if layer_names is None:
        layer_names = lens.adapter.get_patchable_layers()
    # Guaranteed to be a list at this point
    assert layer_names is not None, "No patchable layers found."

    if method == "attribution":
        _patch = run_attribution_patching
    elif method == "exact":
        _patch = run_activation_patching
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'exact' or 'attribution'.")

    patch_results = _patch(
        lens,
        clean_input,
        corrupted_input,
        layer_names=layer_names,
        metric_fn=metric_fn,
        **kwargs,
    )

    # Step 2: Filter to important sublayers
    nodes = _build_nodes(lens, patch_results, layer_names, importance_threshold)

    if not nodes:
        return {
            "nodes": [],
            "edges": [],
            "patch_results": patch_results,
            "attention_results": None,
            "message": "No components exceeded the importance threshold.",
        }

    # Step 3: Get attention patterns for information routing (transformers only)
    attn_results = None
    if lens.adapter.supports(AnalysisCapability.ATTENTION_MAPS):
        attn_results = _safe_attention_analysis(lens, clean_input, **kwargs)

    # Step 4: Build edges from attention patterns and sequential flow
    edges = _build_edges(nodes, attn_results)

    # Step 5: Assign functional roles
    _assign_roles(nodes, patch_results)

    # Sort nodes by layer order
    nodes.sort(key=lambda n: n["order"])

    return {
        "nodes": nodes,
        "edges": edges,
        "patch_results": patch_results,
        "attention_results": attn_results,
        "clean_metric": patch_results["clean_metric"],
        "corrupted_metric": patch_results["corrupted_metric"],
        "total_effect": patch_results["total_effect"],
        "num_components": len(nodes),
        "num_connections": len(edges),
    }


def _build_nodes(
    lens, patch_results: Dict, layer_names: List[str], threshold: float
) -> List[Dict]:
    """Extract important components as circuit nodes."""
    nodes = []

    for i, name in enumerate(layer_names):
        data = patch_results["patch_effects"].get(name, {})
        norm_effect = abs(data.get("normalized_effect", 0.0))

        if norm_effect < threshold:
            continue

        family = lens.adapter.infer_module_family(name)
        block_num = _extract_block_number(name)

        nodes.append(
            {
                "name": name,
                "order": i,
                "block_num": block_num,
                "family": family,
                "normalized_effect": data.get("normalized_effect", 0.0),
                "effect": data.get("effect", 0.0),
                "patched_metric": data.get("patched_metric", 0.0),
                "role": None,  # assigned in _assign_roles
            }
        )

    return nodes


def _build_edges(nodes: List[Dict], attn_results: Optional[Dict]) -> List[Dict]:
    """
    Build directed edges between circuit nodes.

    Two types of edges:
    1. Sequential: components in block N connect to components in
       block N+1 via the residual stream.
    2. Attention-based: attention heads route information between
       positions, connecting to downstream components.
    """
    edges = []

    # Sequential edges between consecutive blocks
    edges.extend(_build_sequential_edges(nodes))

    # Attention-based edges (transformers only)
    if attn_results and "attention_maps" in attn_results:
        edges.extend(_build_attention_edges(nodes, attn_results))

    # Deduplicate
    seen = set()
    unique = []
    for e in edges:
        key = (e["from"], e["to"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def _build_sequential_edges(nodes: List[Dict]) -> List[Dict]:
    """Connect components across consecutive blocks."""
    edges = []
    sorted_nodes = sorted(nodes, key=lambda n: n["order"])

    for i in range(len(sorted_nodes)):
        src = sorted_nodes[i]
        if src["block_num"] is None:
            continue

        for j in range(i + 1, len(sorted_nodes)):
            dst = sorted_nodes[j]
            if dst["block_num"] is None:
                continue

            gap = dst["block_num"] - src["block_num"]
            if 0 < gap <= 2:
                edges.append(
                    {
                        "from": src["name"],
                        "to": dst["name"],
                        "type": "sequential",
                        "weight": min(
                            abs(src["normalized_effect"]),
                            abs(dst["normalized_effect"]),
                        ),
                    }
                )
                break  # Only connect to nearest downstream

    return edges


def _build_attention_edges(nodes: List[Dict], attn_results: Dict) -> List[Dict]:
    """Build edges from attention patterns showing information routing."""
    edges = []
    attn_nodes = [n for n in nodes if n["family"] == "attention"]

    for node in attn_nodes:
        attn_key = node["name"]
        if attn_key not in attn_results["attention_maps"]:
            continue

        weights = attn_results["attention_maps"][attn_key]["weights"]

        # Get average attention at the last position
        if weights.dim() == 4:
            avg_attn = weights[0].mean(dim=0)  # (seq, seq)
        elif weights.dim() == 3:
            avg_attn = weights[0]
        else:
            continue

        last_attn = avg_attn[-1]
        max_attended_pos = int(last_attn.argmax().item())
        max_attn_weight = float(last_attn.max().item())

        # Attention feeds into MLP within the same block
        for other in nodes:
            if other["name"] == node["name"]:
                continue
            if (
                other["block_num"] is not None
                and node["block_num"] is not None
                and other["block_num"] == node["block_num"]
                and other["family"] == "mlp"
            ):
                edges.append(
                    {
                        "from": node["name"],
                        "to": other["name"],
                        "type": "attention_routing",
                        "weight": max_attn_weight,
                        "attended_position": max_attended_pos,
                    }
                )

    return edges


def _assign_roles(nodes: List[Dict], patch_results: Dict) -> None:
    """
    Assign functional roles to circuit nodes.

    Roles:
    - "critical": corrupting this component nearly destroys the behavior
    - "booster": corrupted version actually improves the metric
    - "gate": late-layer component shaping final output
    - "processor": moderate effect, part of the processing pipeline
    """
    clean_metric = patch_results["clean_metric"]

    for node in nodes:
        effect = node["normalized_effect"]
        patched = node["patched_metric"]

        if abs(effect) > 0.7:
            node["role"] = "critical"
        elif effect < -0.3 and patched > clean_metric:
            node["role"] = "booster"
        elif node["block_num"] is not None and _is_late_layer(node, nodes):
            node["role"] = "gate"
        else:
            node["role"] = "processor"


def _is_late_layer(node: Dict, all_nodes: List[Dict]) -> bool:
    """Check if node is in the last 20% of blocks."""
    block_nums = [n["block_num"] for n in all_nodes if n["block_num"] is not None]
    if not block_nums:
        return False
    return node["block_num"] >= max(block_nums) * 0.8


def _extract_block_number(name: str) -> Optional[int]:
    """Extract the block/layer number from a module name."""
    match = re.search(r"\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    match = re.search(r"\.(\d+)$", name)
    if match:
        return int(match.group(1))
    return None


def _safe_attention_analysis(lens, inputs, **kwargs) -> Optional[Dict]:
    """Run attention analysis, returning None if it fails."""
    try:
        from modellens.analysis.attention import run_attention_analysis

        return run_attention_analysis(lens, inputs, **kwargs)
    except Exception:
        return None


def summarize_circuit(circuit: Dict) -> str:
    """Generate a human-readable summary of the discovered circuit."""
    nodes = circuit.get("nodes", [])
    edges = circuit.get("edges", [])

    if not nodes:
        return "No significant circuit components found."

    lines = [
        f"Circuit: {len(nodes)} components, {len(edges)} connections",
        f"Clean metric: {circuit.get('clean_metric', 0):.4f}",
        f"Corrupted metric: {circuit.get('corrupted_metric', 0):.4f}",
        f"Total effect: {circuit.get('total_effect', 0):.4f}",
        "",
    ]

    # Group by role
    for role in ["critical", "booster", "gate", "processor"]:
        role_nodes = [n for n in nodes if n["role"] == role]
        if role_nodes:
            lines.append(f"{role.upper()} ({len(role_nodes)}):")
            for n in role_nodes:
                lines.append(
                    f"  {n['name']} ({n['family']}) — "
                    f"effect: {n['normalized_effect']:+.3f}"
                )
            lines.append("")

    # Attention routing
    attn_edges = [e for e in edges if e["type"] == "attention_routing"]
    if attn_edges:
        lines.append("Attention routing:")
        for e in attn_edges:
            lines.append(
                f"  {e['from']} -> {e['to']} "
                f"(weight: {e['weight']:.3f}, "
                f"attends to pos {e.get('attended_position', '?')})"
            )

    return "\n".join(lines)
