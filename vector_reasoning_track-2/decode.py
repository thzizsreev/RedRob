"""LangVAE decoder — maps 128d latent vectors to text."""

from __future__ import annotations

import numpy as np
import torch

from encode import get_langvae_model


def langvae_decode(z_128: np.ndarray) -> str:
    """Decode a steered 128d latent vector via LangVAE's canonical API."""
    model = get_langvae_model()
    z = torch.tensor(z_128, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        decoded = model.decode_sentences(z)[0]
    return decoded.strip()
