"""GTR-base encoder wrapper (SentenceTransformer)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_ENCODER: SentenceTransformer | None = None
_ENCODER_NAME: str | None = None


def load_encoder(model_name: str) -> SentenceTransformer:
    global _ENCODER, _ENCODER_NAME
    if _ENCODER is not None and _ENCODER_NAME == model_name:
        return _ENCODER

    from sentence_transformers import SentenceTransformer

    _ENCODER = SentenceTransformer(model_name)
    _ENCODER_NAME = model_name
    return _ENCODER


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def encode_texts(model_name: str, texts: list[str]) -> np.ndarray:
    """Encode texts to (N, 768) L2-normalized vectors."""
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    encoder = load_encoder(model_name)
    vectors = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    return _l2_normalize(vectors)
