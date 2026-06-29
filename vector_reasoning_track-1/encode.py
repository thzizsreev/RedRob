"""LangVAE encoder — maps text to 128d μ latent vectors."""

from __future__ import annotations

import numpy as np
import torch
from langvae import LangVAE

LANGVAE_HF_ID = "neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128"
LATENT_DIM = 128

_langvae_model: LangVAE | None = None


def get_langvae_model() -> LangVAE:
    global _langvae_model
    if _langvae_model is None:
        _langvae_model = LangVAE.load_from_hf_hub(LANGVAE_HF_ID)
        _langvae_model.eval()
        _langvae_model.encoder.init_pretrained_model()
    return _langvae_model


def langvae_encode(text: str) -> np.ndarray:
    """Return 128d μ latent from LangVAE's bert-base-cased encoder."""
    model = get_langvae_model()
    input_ids = model.encoder.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )["input_ids"]
    with torch.no_grad():
        mu = model.encoder(input_ids).embedding
    return mu.squeeze().cpu().numpy()
