import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from modellens.adapters.base import AnalysisCapability
from modellens.helpers.activations import gather_activation_rows
from modellens.helpers.layers import default_residual_layer


class SparseAutoencoder(nn.Module):
    """
    Overcomplete sparse autoencoder for dictionary learning on activations.

    Architecture: encoder (Linear d -> h) -> ReLU -> decoder (Linear h -> d),
    with an overcomplete hidden dim h = input_dim * expansion.

    Weights are UNTIED — the encoder and decoder learn independent matrices.
    Tying (decoder = encoder.T) halves the parameter count but empirically
    hurts reconstruction and feature monosemanticity, and the standard SAE
    recipe (Anthropic, "Towards Monosemanticity", 2023) uses untied weights,
    so we follow that here.

    The ReLU on the hidden code guarantees non-negative feature activations,
    and an L1 penalty on that code (applied in train_sae) drives sparsity.
    """

    def __init__(self, input_dim: int, expansion: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.expansion = expansion
        self.num_features = input_dim * expansion
        self.encoder = nn.Linear(input_dim, self.num_features)
        self.decoder = nn.Linear(self.num_features, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map activations to the (non-negative) sparse feature code."""
        return F.relu(self.encoder(x))

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations from the feature code."""
        return self.decoder(code)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.encode(x)
        recon = self.decode(code)
        return recon, code


def train_sae(
    lens,
    inputs: Any,
    *,
    layer_name: Optional[str] = None,
    expansion: int = 4,
    l1_coeff: float = 1e-3,
    steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    **kwargs,
) -> Tuple[SparseAutoencoder, Dict[str, Any]]:
    """
    Fit a sparse autoencoder on the activations ModelLens captures at a layer.

    Activations are gathered through the lens' own hooks (never re-implemented
    here), flattened to a matrix of activation vectors, and used to train an
    overcomplete SAE with an L1 sparsity penalty on the hidden code.

    Args:
        lens: ModelLens instance
        inputs: Model input, or a list of inputs, to source activations from
        layer_name: Layer to hook. If None, defaults to a residual-stream
                    layer when the model exposes one (same lookup as the
                    residual_stream analysis); otherwise required.
        expansion: Overcomplete factor — hidden dim = input_dim * expansion
        l1_coeff: Weight of the L1 sparsity penalty on the hidden code
        steps: Number of gradient steps
        lr: Adam learning rate
        batch_size: Minibatch size (clamped to the number of activations)
        seed: Seed for reproducible initialization/sampling

    Returns:
        (trained SparseAutoencoder, training summary dict with final recon
        loss, L0/L1 sparsity, and dead-feature count).
    """
    layer_name = _resolve_layer(lens, layer_name, "train_sae")
    torch.manual_seed(seed)

    X = _gather_matrix(lens, inputs, layer_name, **kwargs)
    n, d = X.shape

    sae = SparseAutoencoder(d, expansion=expansion)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Baseline reconstruction before any training, for comparison.
    sae.eval()
    with torch.no_grad():
        initial_recon = F.mse_loss(sae(X)[0], X).item()

    sae.train()
    bs = min(batch_size, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))
        batch = X[idx]
        recon, code = sae(batch)
        recon_loss = F.mse_loss(recon, batch)
        # L1 sparsity penalty on the hidden code: sum over features, mean
        # over the batch.
        l1 = code.abs().sum(dim=1).mean()
        loss = recon_loss + l1_coeff * l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final metrics on the full activation matrix.
    sae.eval()
    with torch.no_grad():
        recon, code = sae(X)
        final_recon = F.mse_loss(recon, X).item()
        active = code > 0
        l0 = active.float().sum(dim=1).mean().item()
        l1_val = code.abs().sum(dim=1).mean().item()
        dead_features = int((~active.any(dim=0)).sum().item())

    summary = {
        "layer_name": layer_name,
        "input_dim": d,
        "num_features": sae.num_features,
        "expansion": expansion,
        "num_activations": n,
        "steps": steps,
        "l1_coeff": l1_coeff,
        "initial_recon_loss": initial_recon,
        "final_recon_loss": final_recon,
        "l0_sparsity": l0,
        "l1_sparsity": l1_val,
        "dead_features": dead_features,
    }
    return sae, summary


def sae_features(
    lens,
    inputs: Any,
    sae: SparseAutoencoder,
    *,
    layer_name: Optional[str] = None,
    top_k: int = 10,
    tokenizer=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run inputs through a trained SAE and report which features fire.

    For every input it reports the active features (max-pooled over token
    positions) and the top-k strongest features. It also builds a per-feature
    "top activating inputs/tokens" summary so features are actually
    inspectable — the whole point of an SAE.

    Args:
        lens: ModelLens instance
        inputs: A single input or a list of inputs
        sae: A trained SparseAutoencoder (from train_sae)
        layer_name: Layer to hook. If None, defaults to the same
                    residual-stream layer train_sae would use.
        top_k: Number of top features (per input) and top activations
               (per feature) to report
        tokenizer: Optional tokenizer to attach token strings to activations

    Returns:
        Dict with per-input firings and a per-feature top-activation summary.
    """
    layer_name = _resolve_layer(lens, layer_name, "sae_features")
    sae.eval()

    input_list = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]

    per_input: List[Dict[str, Any]] = []
    # feature index -> list of (value, input_index, position, token)
    feature_hits: Dict[int, List[Tuple[float, int, int, Optional[str]]]] = {}

    for i, inp in enumerate(input_list):
        rows, prov = gather_activation_rows(lens, inp, layer_name, tokenizer, **kwargs)
        with torch.no_grad():
            codes = sae.encode(rows)  # (num_rows, num_features), >= 0

        # Per-input view: max activation per feature across token positions.
        rep = codes.max(dim=0).values
        k = min(top_k, rep.shape[0])
        top_vals, top_idx = torch.topk(rep, k=k)
        active = torch.nonzero(rep > 0, as_tuple=False).flatten().tolist()
        per_input.append(
            {
                "input_index": i,
                "active_features": active,
                "num_active": len(active),
                "top_features": [
                    (int(fi), float(v)) for fi, v in zip(top_idx, top_vals)
                ],
                "activations": rep.detach(),
            }
        )

        # Accumulate every (feature, position) firing for the summary.
        for r in range(codes.shape[0]):
            row = codes[r]
            for f in torch.nonzero(row > 0, as_tuple=False).flatten().tolist():
                feature_hits.setdefault(f, []).append(
                    (float(row[f]), i, prov[r]["position"], prov[r].get("token"))
                )

    # Per-feature top activating inputs/tokens.
    feature_summary: Dict[int, Dict[str, Any]] = {}
    for f, hits in feature_hits.items():
        hits.sort(key=lambda t: t[0], reverse=True)
        feature_summary[f] = {
            "top_activations": [
                {"value": v, "input_index": ii, "position": pos, "token": tok}
                for (v, ii, pos, tok) in hits[:top_k]
            ]
        }

    return {
        "layer_name": layer_name,
        "per_input": per_input,
        "feature_summary": feature_summary,
        "num_features": sae.num_features,
        "num_inputs": len(input_list),
        "num_active_features": len(feature_summary),
    }


def dictionary_features(lens, activations: Any, *, top_k: int = 10) -> Dict[str, Any]:
    """
    Inspect a sparse autoencoder's learned feature dictionary.

    Unlike sae_features (which encodes a *host* model's activations through an
    external SAE probe), this points ModelLens directly at a trained SAE — the
    SAE is the model under the lens — and passes activation vectors in the SAE's
    own input space. It reports which dictionary features fire, per-feature
    activation statistics, dead features, and the decoder directions (the
    dictionary) each feature writes.

    Available when the adapter detects an overcomplete autoencoder
    (DICTIONARY_ANALYSIS capability).

    Args:
        lens: ModelLens wrapping a trained SAE
        activations: Activation vectors, shape (N, input_dim)
        top_k: Number of top features to report per input

    Returns:
        Dict with per-input firings, per-feature stats, dead features, and the
        decoder dictionary norms.
    """
    lens.adapter.require(AnalysisCapability.DICTIONARY_ANALYSIS, "dictionary_features")

    model = lens.model
    X = activations if isinstance(activations, torch.Tensor) else torch.as_tensor(activations)
    X = X.float()
    if X.dim() == 1:
        X = X.unsqueeze(0)

    with torch.no_grad():
        if hasattr(model, "encode"):
            codes = model.encode(X)
        else:
            # Fallback: first Linear followed by ReLU.
            linear = next(m for m in model.modules() if isinstance(m, nn.Linear))
            codes = F.relu(linear(X))

    num_features = codes.shape[1]
    fired = codes > 0
    fires_per_feature = fired.sum(dim=0)
    dead = torch.nonzero(fires_per_feature == 0, as_tuple=False).flatten().tolist()
    active = torch.nonzero(fires_per_feature > 0, as_tuple=False).flatten().tolist()

    # Decoder directions = the learned dictionary (last Linear's columns).
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    decoder_norms = None
    if linears and linears[-1].in_features == num_features:
        decoder_norms = linears[-1].weight.detach().norm(dim=0)  # (num_features,)

    feature_stats = {}
    for f in active:
        col = codes[:, f]
        stat = {
            "activation_mean": float(col.mean()),
            "activation_max": float(col.max()),
            "fires": int(fires_per_feature[f].item()),
        }
        if decoder_norms is not None:
            stat["direction_norm"] = float(decoder_norms[f].item())
        feature_stats[f] = stat

    per_input = []
    for i in range(X.shape[0]):
        row = codes[i]
        k = min(top_k, num_features)
        top_vals, top_idx = torch.topk(row, k=k)
        per_input.append(
            {
                "input_index": i,
                "num_active": int((row > 0).sum().item()),
                "top_features": [
                    (int(fi), float(v)) for fi, v in zip(top_idx, top_vals)
                ],
            }
        )

    return {
        "num_features": num_features,
        "num_inputs": int(X.shape[0]),
        "active_features": active,
        "dead_features": dead,
        "feature_stats": feature_stats,
        "per_input": per_input,
    }


def feature_directions(lens, *, normalize: bool = False) -> Dict[str, Any]:
    """
    Return a sparse autoencoder's learned dictionary — the direction each
    feature writes into activation space.

    This is the third view of an SAE, alongside:
      - sae_features:       what features mean (max-activating inputs/tokens)
      - dictionary_features: SAE health (dead features, firing statistics)
      - feature_directions:  the dictionary itself (the decoder vectors)

    Point ModelLens at a trained SAE. Each feature's direction is the matching
    column of the decoder weight (what the feature adds to the reconstruction);
    the encoder rows (what each feature reads in) are returned too when they
    line up with the feature count.

    Args:
        lens: ModelLens wrapping a trained SAE
        normalize: If True, return unit-norm direction vectors

    Returns:
        Dict with the decoder directions (num_features, input_dim), their
        norms, and — when available — the encoder read-in directions.
    """
    lens.adapter.require(AnalysisCapability.DICTIONARY_ANALYSIS, "feature_directions")

    model = lens.model
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        raise ValueError(
            "Model does not look like an autoencoder (need >= 2 Linear layers)."
        )

    encoder, decoder = linears[0], linears[-1]
    # decoder.weight is (input_dim, num_features); feature j's direction is
    # column j — transpose so row j is that direction.
    directions = decoder.weight.detach().t().contiguous()  # (num_features, input_dim)
    norms = directions.norm(dim=1)  # (num_features,)

    if normalize:
        directions = directions / (norms.unsqueeze(1) + 1e-10)

    result = {
        "num_features": directions.shape[0],
        "input_dim": directions.shape[1],
        "directions": directions,
        "norms": norms,
    }

    # Encoder rows (read-in directions) — only when they match the feature count.
    if encoder.out_features == directions.shape[0]:
        result["encoder_directions"] = encoder.weight.detach().contiguous()

    return result


# ---- Private Helpers ----


def _resolve_layer(lens, layer_name: Optional[str], analysis_name: str) -> str:
    """
    Require activation-capture support and resolve the hook site.

    SAE only needs activation capture, which every adapter exposes via the
    HOOKS capability. When no layer_name is given, default to a residual-stream
    layer if the model has one.
    """
    lens.adapter.require(AnalysisCapability.HOOKS, analysis_name)
    if layer_name is not None:
        return layer_name

    default = default_residual_layer(lens)
    if default is None:
        raise ValueError(
            "No layer_name given and the model has no residual stream to "
            "default to. Pass layer_name explicitly."
        )
    return default


def _gather_matrix(lens, inputs: Any, layer_name: str, **kwargs) -> torch.Tensor:
    """Collect activations at layer_name across inputs into an (N, d) matrix."""
    input_list = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
    chunks = []
    for inp in input_list:
        rows, _ = gather_activation_rows(lens, inp, layer_name, None, **kwargs)
        chunks.append(rows)
    if not chunks:
        raise ValueError("No activations gathered — empty inputs.")
    return torch.cat(chunks, dim=0)
