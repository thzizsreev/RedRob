"""GTR-base encoder — single forward pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_ENCODER: SentenceTransformer | None = None


def _bootstrap_imports() -> None:
    import pyarrow  # noqa: F401
    import pyarrow.dataset  # noqa: F401
    import datasets  # noqa: F401


_bootstrap_imports()


def encode_text(model_name: str, text: str) -> np.ndarray:
    """Encode one sentence to a (768,) L2-normalized vector."""
    global _ENCODER
    from sentence_transformers import SentenceTransformer

    if _ENCODER is None:
        _ENCODER = SentenceTransformer(model_name)

    vector = _ENCODER.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector = vector / norm
    return vector
