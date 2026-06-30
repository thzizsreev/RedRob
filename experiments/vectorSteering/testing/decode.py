"""Vec2Text decoder wrapper for GTR-base embeddings."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

_CORRECTOR: Any | None = None
_CORRECTOR_ID: str | None = None


def _ensure_windows_resource_stub() -> None:
    """vec2text imports Unix-only `resource` at package load time."""
    if sys.platform == "win32" and "resource" not in sys.modules:
        stub = types.ModuleType("resource")
        stub.RLIMIT_NOFILE = 7

        def _getrlimit(_which: int) -> tuple[int, int]:
            return (8192, 8192)

        stub.getrlimit = _getrlimit
        stub.setrlimit = lambda *_args, **_kwargs: None
        sys.modules["resource"] = stub


def _prepare_torch() -> None:
    import torch

    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")


def load_corrector(embedder_id: str = "gtr-base") -> Any:
    global _CORRECTOR, _CORRECTOR_ID
    if _CORRECTOR is not None and _CORRECTOR_ID == embedder_id:
        return _CORRECTOR

    _ensure_windows_resource_stub()
    _prepare_torch()
    import vec2text

    _CORRECTOR = vec2text.load_pretrained_corrector(embedder_id)
    _CORRECTOR_ID = embedder_id
    return _CORRECTOR


def decode_vectors(
    vectors: np.ndarray,
    *,
    embedder_id: str = "gtr-base",
    num_steps: int = 20,
    sequence_beam_width: int = 4,
) -> list[str]:
    """Invert (N, 768) embedding rows back to text strings."""
    import torch

    _ensure_windows_resource_stub()
    _prepare_torch()
    import vec2text

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    corrector = load_corrector(embedder_id)
    embeddings = torch.from_numpy(vectors.astype(np.float32))

    texts = vec2text.invert_embeddings(
        embeddings=embeddings,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )
    return [str(t) for t in texts]
