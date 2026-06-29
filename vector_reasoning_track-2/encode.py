"""SONAR text encoder — maps text to 1024d vectors."""

from __future__ import annotations

import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

from constants import SONAR_ENCODER, SOURCE_LANG

_encoder: TextToEmbeddingModelPipeline | None = None


def get_encoder() -> TextToEmbeddingModelPipeline:
    global _encoder
    if _encoder is None:
        _encoder = TextToEmbeddingModelPipeline(
            encoder=SONAR_ENCODER,
            tokenizer=SONAR_ENCODER,
        )
    return _encoder


def sonar_encode(text: str) -> np.ndarray:
    """Encode one string into a 1024d SONAR vector."""
    embedding = get_encoder().predict([text], source_lang=SOURCE_LANG)
    return embedding[0].numpy()
