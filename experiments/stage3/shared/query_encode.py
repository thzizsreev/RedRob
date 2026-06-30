"""ONNX query encoding for Stage 3 precompute."""

from __future__ import annotations

import faiss
import numpy as np

from tracks.instructor.core.config import (
    EVAL_INSTRUCTION,
    INFRA_INSTRUCTION,
    RETRIEVAL_INSTRUCTION,
    VECTOR_DIM,
)
from experiments.stage3.shared.config_precompute import PrecomputeConfig, SubspaceWeights
from experiments.stage3.shared.cpu_embedder import InstructorONNX


def _normalize_single_block(vec: np.ndarray) -> np.ndarray:
    arr = vec.reshape(1, -1).astype(np.float32).copy()
    faiss.normalize_L2(arr)
    return arr.reshape(-1)


def encode_weighted_query(
    model: InstructorONNX,
    text: str,
    weights: SubspaceWeights,
) -> np.ndarray:
    instructions = (RETRIEVAL_INSTRUCTION, INFRA_INSTRUCTION, EVAL_INSTRUCTION)
    weight_tuple = weights.as_tuple()

    blocks: list[np.ndarray] = []
    for instruction, weight in zip(instructions, weight_tuple):
        raw = model.encode(
            [[instruction, text]], batch_size=1, normalize=False
        ).reshape(-1)
        normalized = _normalize_single_block(raw)
        blocks.append((normalized * weight).astype(np.float32))

    vec = np.concatenate(blocks).astype(np.float32)
    if vec.shape != (VECTOR_DIM,):
        raise ValueError(f"Expected query vector shape ({VECTOR_DIM},), got {vec.shape}")
    return vec


def encode_stage3_queries(
    model: InstructorONNX,
    config: PrecomputeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = encode_weighted_query(model, config.q1_text, config.subspace_weights_q1)
    q2 = encode_weighted_query(model, config.q2_text, config.subspace_weights_q2)
    q3 = encode_weighted_query(model, config.q3_text, config.subspace_weights_q3)
    return q1, q2, q3
