"""Stage 9: Weighted query vector construction for rank-time retrieval."""

from __future__ import annotations

import numpy as np

from pipeline.config import (
    QUERY_EVAL_TEXT,
    QUERY_INFRA_TEXT,
    QUERY_RETRIEVAL_TEXT,
    QUERY_WEIGHTS,
)


def _encode_query_subvector(model, text: str) -> np.ndarray:
    vec = model.encode(text, normalize_embeddings=False)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def build_query_vector(
    model,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
) -> np.ndarray:
    """
    Build weighted query vector from fixed JD-derived query texts.
    Weights applied before concatenation: [w_r * q_r | w_i * q_i | w_e * q_e].
    """
    v_r = _encode_query_subvector(model, QUERY_RETRIEVAL_TEXT)
    v_i = _encode_query_subvector(model, QUERY_INFRA_TEXT)
    v_e = _encode_query_subvector(model, QUERY_EVAL_TEXT)

    w_r, w_i, w_e = weights
    return np.concatenate([w_r * v_r, w_i * v_i, w_e * v_e]).astype(np.float32)
