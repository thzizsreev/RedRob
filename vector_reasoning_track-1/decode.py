"""GPT-2 decode via LangVAE — maps 128d latent vectors to text."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from constants import DO_SAMPLE, MAX_LENGTH, TEMPERATURE, TOP_P
from encode import get_langvae_model


def gpt2_decode(
    z_128: np.ndarray,
    max_length: int = MAX_LENGTH,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    do_sample: bool = DO_SAMPLE,
) -> str:
    """Decode a steered 128d latent vector via LangVAE's GPT-2 decoder."""
    model = get_langvae_model()
    z = torch.tensor(z_128, dtype=torch.float32).unsqueeze(0)

    decoder = model.decoder
    generate_fn = getattr(decoder, "generate", None)
    if generate_fn is not None:
        sig = inspect.signature(generate_fn)
        kwargs: dict = {"z": z}
        if "max_length" in sig.parameters:
            kwargs["max_length"] = max_length
        elif "max_new_tokens" in sig.parameters:
            kwargs["max_new_tokens"] = max_length
        if "temperature" in sig.parameters:
            kwargs["temperature"] = temperature
        if "top_p" in sig.parameters:
            kwargs["top_p"] = top_p
        if "do_sample" in sig.parameters:
            kwargs["do_sample"] = do_sample

        with torch.no_grad():
            output = generate_fn(**kwargs)

        if isinstance(output, torch.Tensor):
            tokenizer = decoder.tokenizer
            return tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if isinstance(output, (list, tuple)) and output:
            first = output[0]
            if isinstance(first, str):
                return first.strip()

    with torch.no_grad():
        decoded = model.decode_sentences(z)[0]
    return decoded.strip()
