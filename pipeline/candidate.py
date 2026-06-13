"""Stage 7: Full per-candidate vector encoding pipeline."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.assignment import assign_sentences_to_blocks, encode_sentences
from pipeline.block_text import build_block_text
from pipeline.encoding import assemble_candidate_vector, encode_block
from pipeline.extraction import build_candidate_segments
from pipeline.model_utils import embedding_dim
from pipeline.sentences import split_into_sentences


def process_one_candidate(
    record: dict,
    model: SentenceTransformer,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> tuple[str, np.ndarray]:
    """Run stages 1-7 for one candidate. Returns (candidate_id, 1152-d vector)."""
    candidate_id = record["candidate_id"]
    dim = embedding_dim(model)
    null_vector = np.zeros(dim * 3, dtype=np.float32)

    segments = build_candidate_segments(record)
    if not segments:
        return candidate_id, null_vector

    sentences = split_into_sentences(segments)
    if not sentences:
        return candidate_id, null_vector

    sentence_vecs = encode_sentences(sentences, model)
    block_assignments = assign_sentences_to_blocks(
        sentences, sentence_vecs, anchors, thresholds
    )

    block_texts = {
        block: build_block_text(assignments)
        for block, assignments in block_assignments.items()
    }

    v_r = encode_block(block_texts["retrieval"], model)
    v_i = encode_block(block_texts["infra"], model)
    v_e = encode_block(block_texts["eval"], model)

    return candidate_id, assemble_candidate_vector(v_r, v_i, v_e)
