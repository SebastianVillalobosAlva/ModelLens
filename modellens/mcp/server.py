"""
MCP server wrapping ModelLens interpretability analyses.

Exposes a curated set of analyses as MCP tools so they can be driven
conversationally. Each tool loads a small model by reference, runs one
analysis, and returns JSON-safe output (tensors are converted to lists or
compact summaries).

Run it:
    modellens-mcp

Requires the optional MCP dependency:
    pip install "modellens[mcp]"
"""

from typing import Any, Dict, Optional

import torch

from mcp.server.fastmcp import FastMCP

from modellens.core.lens import ModelLens
from modellens.analysis.sparse_autoencoder import train_sae
from modellens.helpers.activations import gather_activation_rows

server = FastMCP("modellens")

# Cache loaded (lens, tokenizer) pairs so repeated tool calls on the same
# model don't reload weights.
_LENS_CACHE: Dict[str, Any] = {}


# ---- Model loading (minimal — assumes small HF causal-LM models) ----


def _load_lens(model_ref: str):
    """
    Load a small HuggingFace causal-LM by name (e.g. "gpt2",
    "sshleifer/tiny-gpt2") and wrap it in a ModelLens with its tokenizer
    attached, so raw-string inputs are tokenized automatically.
    """
    if model_ref in _LENS_CACHE:
        return _LENS_CACHE[model_ref]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(model_ref)
    lens = ModelLens(model)
    if hasattr(lens.adapter, "set_tokenizer"):
        lens.adapter.set_tokenizer(tokenizer)

    _LENS_CACHE[model_ref] = (lens, tokenizer)
    return lens, tokenizer


# ---- JSON-safe conversion ----


def _to_jsonable(obj: Any, tensor_cap: int = 32) -> Any:
    """
    Recursively convert an analysis result into something JSON-serializable.
    Small tensors become lists; large tensors become a shape/stats summary so
    nothing that won't serialize leaks out.
    """
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if t.numel() <= tensor_cap:
            return t.reshape(-1).tolist() if t.dim() > 0 else t.item()
        tf = t.float()
        return {
            "_tensor": True,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "mean": float(tf.mean()),
            "min": float(tf.min()),
            "max": float(tf.max()),
        }
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, tensor_cap) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, tensor_cap) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---- Tools ----


@server.tool()
def logit_lens(model_ref: str, text: str, top_k: int = 5) -> dict:
    """Project each layer's hidden state to output space; return per-layer
    top-k predicted tokens for the given text."""
    lens, tokenizer = _load_lens(model_ref)
    result = lens.layer_probe(text, top_k=top_k)

    layers = {}
    for name, data in result["layer_results"].items():
        idx = data["top_k_indices"][0]
        probs = data["top_k_probs"][0]
        layers[name] = [
            [tokenizer.decode([i.item()]).strip(), round(p.item(), 4)]
            for i, p in zip(idx, probs)
        ]
    return {"model": model_ref, "num_layers": len(layers), "layers": layers}


@server.tool()
def layer_evolution(model_ref: str, text: str, top_k: int = 10) -> dict:
    """Track how the prediction distribution (top-k, entropy, KL) evolves
    across layers for the given text."""
    lens, tokenizer = _load_lens(model_ref)
    result = lens.layer_evolution(text, top_k=top_k, tokenizer=tokenizer)
    return _to_jsonable(result)


@server.tool()
def discover_circuit(
    model_ref: str, clean_text: str, corrupted_text: str
) -> dict:
    """Discover the causal circuit distinguishing clean vs corrupted input
    via activation patching, attention flow, and role assignment."""
    lens, _ = _load_lens(model_ref)
    result = lens.discover_circuit(clean_text, corrupted_text)
    return _to_jsonable(result)


@server.tool()
def sae_analysis(
    model_ref: str,
    text: str,
    layer_name: Optional[str] = None,
    expansion: int = 4,
    steps: int = 200,
    top_k: int = 10,
) -> dict:
    """Train a sparse autoencoder on the model's activations for the given text,
    then return three consistent views of that one SAE: what its top features
    mean (max-activating tokens), its health (dead vs. active features), and its
    learned dictionary (top features by decoder-vector norm). All views share
    the same feature indices because they come from a single trained SAE."""
    lens, tokenizer = _load_lens(model_ref)
    sae, summary = train_sae(
        lens, text, layer_name=layer_name, expansion=expansion, steps=steps
    )
    resolved_layer = summary["layer_name"]

    # View 1 — what features mean (host activations through the SAE probe).
    feats = lens.sae_features(
        text, sae, layer_name=resolved_layer, top_k=top_k, tokenizer=tokenizer
    )
    top_features = [
        {
            "feature": int(feature),
            "peak_activation": round(data["top_activations"][0]["value"], 4),
            "example_token": data["top_activations"][0]["token"],
        }
        for feature, data in feats["feature_summary"].items()
    ]
    top_features.sort(key=lambda f: f["peak_activation"], reverse=True)
    top_features = top_features[:top_k]

    # Views 2 & 3 — point the lens at the trained SAE itself.
    sae_lens = ModelLens(sae)
    rows, _ = gather_activation_rows(lens, text, resolved_layer)
    health = sae_lens.dictionary_features(rows, top_k=top_k)
    directions = sae_lens.feature_directions()

    norms = directions["norms"]
    k = min(top_k, int(norms.shape[0]))
    top_norm_vals, top_norm_idx = torch.topk(norms, k=k)
    top_by_norm = [
        [int(i), round(float(v), 4)] for i, v in zip(top_norm_idx, top_norm_vals)
    ]

    return {
        "model": model_ref,
        "layer": resolved_layer,
        "training": _to_jsonable(summary),
        "top_features": top_features,
        "health": {
            "num_features": health["num_features"],
            "dead": len(health["dead_features"]),
            "active": len(health["active_features"]),
        },
        "dictionary": {
            "num_features": directions["num_features"],
            "input_dim": directions["input_dim"],
            "top_features_by_norm": top_by_norm,
        },
    }


def main() -> None:
    """Console entry point — starts the MCP server over stdio."""
    server.run()


if __name__ == "__main__":
    main()
