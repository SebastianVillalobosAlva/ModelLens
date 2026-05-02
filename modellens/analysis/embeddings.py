import torch
import torch.nn.functional as F
from typing import Dict, Optional


def run_embeddings_analysis(lens, inputs, **kwargs) -> Dict:
    """
    Analyze the embedding representations of the input.

    Args:
        lens: ModelLens instance
        inputs: Model input (string, tensor, or dict)

    Returns:
        Dict with embedding vectors, norms, and similarity data
    """
    embeddings = _get_input_embeddings(lens, inputs, **kwargs)

    if embeddings is None:
        raise ValueError("Could not extract embeddings from the model.")

    # Compute per-token embedding norms
    norms = torch.norm(embeddings, dim=-1)

    # Compute pairwise cosine similarity between token embeddings
    similarity = _cosine_similarity_matrix(embeddings[0])

    return {
        "embeddings": embeddings,  # (batch, seq_len, embed_dim)
        "norms": norms,  # (batch, seq_len)
        "similarity_matrix": similarity,  # (seq_len, seq_len)
        "embed_dim": embeddings.shape[-1],
        "seq_length": embeddings.shape[1],
    }


def _get_input_embeddings(lens, inputs, **kwargs) -> Optional[torch.Tensor]:
    """Extract input embeddings using the adapter's embedding layer."""
    # Try adapter's get_embedding_layer() first
    embed_layer = lens.adapter.get_embedding_layer()

    if embed_layer is not None:
        # Resolve input_ids from different input formats
        input_ids = _resolve_input_ids(lens, inputs)
        if input_ids is not None:
            with torch.no_grad():
                return embed_layer(input_ids)

    return None


def _resolve_input_ids(lens, inputs) -> Optional[torch.Tensor]:
    """
    Extract input_ids from various input formats.
    Handles strings, dicts, and raw tensors.
    """
    if isinstance(inputs, str):
        tokens = lens.adapter.tokenize(inputs)
        return tokens["input_ids"]

    if isinstance(inputs, dict):
        if "input_ids" in inputs:
            return inputs["input_ids"]
        if "input" in inputs:
            return inputs["input"]

    if hasattr(inputs, "input_ids"):
        return inputs["input_ids"]

    if isinstance(inputs, torch.Tensor):
        return inputs

    return None


def _cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between all token positions.

    Args:
        embeddings: (seq_len, embed_dim) tensor

    Returns:
        (seq_len, seq_len) similarity matrix
    """
    normalized = F.normalize(embeddings, dim=-1)
    return normalized @ normalized.T


def nearest_neighbors(lens, token_embedding: torch.Tensor, top_k: int = 10) -> Dict:
    """
    Find the nearest tokens in embedding space to a given embedding vector.

    Args:
        lens: ModelLens instance
        token_embedding: (embed_dim,) vector to find neighbors for
        top_k: Number of nearest neighbors

    Returns:
        Dict with nearest token indices and their similarity scores
    """
    embed_layer = lens.adapter.get_embedding_layer()
    if embed_layer is None:
        raise ValueError("Could not find embedding layer.")

    embed_matrix = embed_layer.weight.detach()

    # Compute cosine similarity against all tokens
    similarity = F.cosine_similarity(token_embedding.unsqueeze(0), embed_matrix, dim=-1)

    top_scores, top_indices = torch.topk(similarity, k=top_k)

    return {
        "indices": top_indices,
        "scores": top_scores,
    }
