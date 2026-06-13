"""Stage 3: Threshold-based soft multi-block sentence assignment."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import SENTENCE_ENCODE_BATCH_SIZE, THRESHOLDS


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def encode_sentences(sentences: list[str], model: SentenceTransformer) -> np.ndarray:
    """Batch encode candidate sentences with passage: prefix."""
    prefixed = [f"passage: {s}" for s in sentences]
    vecs = model.encode(
        prefixed,
        batch_size=SENTENCE_ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return vecs.astype(np.float32)


def assign_sentences_to_blocks(
    sentences: list[str],
    sentence_vecs: np.ndarray,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Soft multi-block assignment via per-anchor cosine similarity."""
    thresholds = thresholds or THRESHOLDS
    blocks: dict[str, list[tuple[str, float]]] = {
        "retrieval": [],
        "infra": [],
        "eval": [],
    }

    for sentence, vec in zip(sentences, sentence_vecs):
        sim_r = cosine_similarity(vec, anchors["retrieval"])
        sim_i = cosine_similarity(vec, anchors["infra"])
        sim_e = cosine_similarity(vec, anchors["eval"])

        if sim_r >= thresholds["retrieval"]:
            blocks["retrieval"].append((sentence, sim_r))
        if sim_i >= thresholds["infra"]:
            blocks["infra"].append((sentence, sim_i))
        if sim_e >= thresholds["eval"]:
            blocks["eval"].append((sentence, sim_e))

    return blocks
