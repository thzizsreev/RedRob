"""SONAR embedding decoder — maps 1024d vectors to text."""

from __future__ import annotations

import numpy as np
import torch
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

from constants import MAX_SEQ_LEN, SONAR_DECODER, SONAR_ENCODER, TARGET_LANG

_decoder: EmbeddingToTextModelPipeline | None = None


def get_decoder() -> EmbeddingToTextModelPipeline:
    global _decoder
    if _decoder is None:
        _decoder = EmbeddingToTextModelPipeline(
            decoder=SONAR_DECODER,
            tokenizer=SONAR_ENCODER,
        )
    return _decoder


def sonar_decode(vector_1024d: np.ndarray) -> str:
    """Decode a 1024d SONAR vector into English text (beam search, deterministic)."""
    t = torch.tensor(vector_1024d, dtype=torch.float32).unsqueeze(0)
    result = get_decoder().predict(
        t,
        target_lang=TARGET_LANG,
        max_seq_len=MAX_SEQ_LEN,
    )
    return result[0]
