"""Vec2Text decoder — single inversion pass (no corrector loop)."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

_CORRECTOR: Any | None = None


def _ensure_windows_resource_stub() -> None:
    if sys.platform == "win32" and "resource" not in sys.modules:
        stub = types.ModuleType("resource")
        stub.RLIMIT_NOFILE = 7
        stub.getrlimit = lambda _which: (8192, 8192)
        stub.setrlimit = lambda *_args, **_kwargs: None
        sys.modules["resource"] = stub


def _inference_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_corrector(embedder_id: str) -> Any:
    global _CORRECTOR
    if _CORRECTOR is not None:
        return _CORRECTOR

    _ensure_windows_resource_stub()
    import torch
    import vec2text
    import vec2text.models.model_utils as model_utils

    device_name = _inference_device()
    model_utils.device = torch.device(device_name)

    corrector = vec2text.load_pretrained_corrector(embedder_id)
    corrector.model.to(device_name).eval()
    corrector.inversion_trainer.model.to(device_name).eval()
    _CORRECTOR = corrector
    return _CORRECTOR


def decode_vector(vector: np.ndarray, *, embedder_id: str = "gtr-base") -> str:
    """Invert one (768,) embedding with a single inversion generate pass."""
    _ensure_windows_resource_stub()
    import torch
    import vec2text

    device_name = _inference_device()

    corrector = _load_corrector(embedder_id)
    embeddings = torch.from_numpy(vector.reshape(1, -1).astype(np.float32)).to(device_name)

    texts = vec2text.invert_embeddings(
        embeddings=embeddings,
        corrector=corrector,
        num_steps=None,
        sequence_beam_width=0,
    )
    return str(texts[0])
