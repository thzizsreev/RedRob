"""Model loading and device helpers for the encoding pipeline."""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from pipeline.config import ENCODE_DEVICE, MODEL_NAME


def embedding_dim(model) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def resolve_device(preference: str | None = None) -> str:
    """Resolve encode device: auto uses cuda when available."""
    preference = (preference or ENCODE_DEVICE).lower()
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference in ("cuda", "cpu"):
        if preference == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return preference
    raise ValueError(f"Unsupported ENCODE_DEVICE: {preference}")


def load_sentence_transformer(
    model_name: str = MODEL_NAME,
    device: str | None = None,
) -> SentenceTransformer:
    resolved = device or resolve_device()
    model = SentenceTransformer(model_name, device=resolved)
    print(f"  device: {resolved}")
    if resolved == "cuda" and torch.cuda.is_available():
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    return model


def model_on_cuda(model: SentenceTransformer) -> bool:
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        return False
    return device.startswith("cuda")
