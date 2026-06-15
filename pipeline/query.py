"""Stage 9: Weighted query vector construction for rank-time retrieval."""

from __future__ import annotations

import numpy as np

from pipeline.config import (
    EVAL_ANCHOR_SENTENCES,
    INFRA_ANCHOR_SENTENCES,
    QUERY_WEIGHTS,
    RETRIEVAL_ANCHOR_SENTENCES,
)
from pipeline.model_utils import embedding_dim


def _encode_query_subvector(model, sentences: list[str] | str) -> np.ndarray:
    if isinstance(sentences, str):
        sentences = [sentences]

    prefixed = [
        s if s.startswith("query:") else f"query: {s}"
        for s in sentences
    ]
    vecs = np.asarray(model.encode(prefixed, normalize_embeddings=False), dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)

    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return np.zeros(embedding_dim(model), dtype=np.float32)
    return (centroid / norm).astype(np.float32)


def build_query_vector(
    model,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
) -> np.ndarray:
    """
    Build weighted query vector from JD anchor sentences.
    Weights applied before concatenation: [w_r * q_r | w_i * q_i | w_e * q_e].
    """
    v_r = _encode_query_subvector(model, RETRIEVAL_ANCHOR_SENTENCES)
    v_i = _encode_query_subvector(model, INFRA_ANCHOR_SENTENCES)
    v_e = _encode_query_subvector(model, EVAL_ANCHOR_SENTENCES)

    w_r, w_i, w_e = weights
    return np.concatenate([w_r * v_r, w_i * v_i, w_e * v_e]).astype(np.float32)


def build_query_vector_from_text(
    model,
    query_text: str,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
) -> np.ndarray:
    """Encode free-text query into the 3-block search vector (same text per block)."""
    vec = _encode_query_subvector(model, query_text)
    w_r, w_i, w_e = weights
    return np.concatenate([w_r * vec, w_i * vec, w_e * vec]).astype(np.float32)
