"""Vec2Text decoder wrapper for GTR-base embeddings."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

_CORRECTOR: Any | None = None
_CORRECTOR_ID: str | None = None
_INFERENCE_DEVICE: str | None = None


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


def _inference_device() -> str:
    global _INFERENCE_DEVICE
    if _INFERENCE_DEVICE is not None:
        return _INFERENCE_DEVICE

    import torch

    if torch.cuda.is_available():
        _INFERENCE_DEVICE = "cuda"
    else:
        _INFERENCE_DEVICE = "cpu"
    return _INFERENCE_DEVICE


def _sync_vec2text_device(device_name: str) -> None:
    import torch
    import vec2text.models.model_utils as model_utils

    model_utils.device = torch.device(device_name)


def _corrector_to_device(corrector: Any, device_name: str) -> Any:
    import torch

    device = torch.device(device_name)
    corrector.model.to(device)
    corrector.inversion_trainer.model.to(device)
    corrector.model.eval()
    corrector.inversion_trainer.model.eval()
    torch.set_grad_enabled(False)
    return corrector


def load_corrector(embedder_id: str = "gtr-base") -> Any:
    global _CORRECTOR, _CORRECTOR_ID
    if _CORRECTOR is not None and _CORRECTOR_ID == embedder_id:
        return _CORRECTOR

    _ensure_windows_resource_stub()
    device_name = _inference_device()
    _sync_vec2text_device(device_name)
    import vec2text

    _CORRECTOR = _corrector_to_device(
        vec2text.load_pretrained_corrector(embedder_id),
        device_name,
    )
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
    device_name = _inference_device()
    _sync_vec2text_device(device_name)
    import vec2text

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    corrector = load_corrector(embedder_id)
    embeddings = torch.from_numpy(vectors.astype(np.float32)).to(device_name)

    texts = vec2text.invert_embeddings(
        embeddings=embeddings,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )
    return [str(t) for t in texts]
