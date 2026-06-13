"""Stages 5-6: Block encoding and candidate vector assembly."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.model_utils import embedding_dim


def encode_block(block_text: str, model: SentenceTransformer) -> np.ndarray:
    """Encode one block's text with passage: prefix and L2 normalize."""
    vec = model.encode(f"passage: {block_text}", normalize_embeddings=False)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros(embedding_dim(model), dtype=np.float32)
    return (vec / norm).astype(np.float32)


def assemble_candidate_vector(
    v_retrieval: np.ndarray,
    v_infra: np.ndarray,
    v_eval: np.ndarray,
) -> np.ndarray:
    """Concatenate three L2-normalized sub-vectors without global normalization."""
    return np.concatenate([v_retrieval, v_infra, v_eval]).astype(np.float32)
