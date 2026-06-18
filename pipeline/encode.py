"""INSTRUCTOR-large 3-pass encoding for Track A precompute (never import from rank.py)."""

from __future__ import annotations

import faiss
import numpy as np
from transformers import AutoTokenizer

from pipeline.config import (
    BLOCK_DIM,
    EVAL_INSTRUCTION,
    INFRA_INSTRUCTION,
    INSTRUCTOR_ONNX_TOKENIZER,
    JD_EVAL_TEXT,
    JD_INFRA_TEXT,
    JD_RETRIEVAL_TEXT,
    ONNX_BATCH_SIZE,
    ONNX_BATCH_SIZE_FALLBACK,
    QUERY_WEIGHTS,
    RETRIEVAL_INSTRUCTION,
    VECTOR_DIM,
)
from pipeline.instructor_onnx import InstructorONNX

def _batch_sizes_to_try(primary: int) -> list[int]:
    seen: set[int] = set()
    sizes: list[int] = []
    for size in (primary, *ONNX_BATCH_SIZE_FALLBACK):
        if size not in seen:
            seen.add(size)
            sizes.append(size)
    return sizes


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).upper()
    if "OUT OF MEMORY" in msg:
        return True
    if "BFCARENA" in msg.replace("_", ""):
        return True
    if "AVAILABLE MEMORY" in msg and "SMALLER THAN REQUESTED" in msg:
        return True
    if "CUDA" in msg and "MEM" in msg:
        return True
    return False


def load_tokenizer():
    return AutoTokenizer.from_pretrained(str(INSTRUCTOR_ONNX_TOKENIZER))


def encode_instruction(
    model: InstructorONNX,
    instruction: str,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    pairs = [[instruction, t] for t in texts]
    last_error: Exception | None = None
    for size in _batch_sizes_to_try(batch_size):
        try:
            if size != batch_size:
                print(f"  Retrying encode with batch_size={size}")
            return model.encode(pairs, batch_size=size, normalize=False)
        except Exception as exc:
            if _is_oom_error(exc):
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise RuntimeError("ONNX encode OOM at minimum batch size") from last_error
    raise RuntimeError("ONNX encode failed")


def normalize_by_block(vecs: np.ndarray, block_size: int = BLOCK_DIM) -> np.ndarray:
    n_blocks = vecs.shape[1] // block_size
    normalized_blocks: list[np.ndarray] = []
    for i in range(n_blocks):
        block = vecs[:, i * block_size : (i + 1) * block_size].copy()
        faiss.normalize_L2(block)
        normalized_blocks.append(block)
    return np.concatenate(normalized_blocks, axis=1).astype(np.float32)


def _normalize_single_block(vec: np.ndarray) -> np.ndarray:
    arr = vec.reshape(1, -1).astype(np.float32).copy()
    faiss.normalize_L2(arr)
    return arr.reshape(-1)


def encode_candidates(
    model: InstructorONNX,
    passages: list[str],
    *,
    batch_size: int = ONNX_BATCH_SIZE,
) -> np.ndarray:
    retrieval = encode_instruction(model, RETRIEVAL_INSTRUCTION, passages, batch_size)
    infra = encode_instruction(model, INFRA_INSTRUCTION, passages, batch_size)
    eval_vecs = encode_instruction(model, EVAL_INSTRUCTION, passages, batch_size)
    combined = np.concatenate([retrieval, infra, eval_vecs], axis=1)
    return normalize_by_block(combined)


def build_jd_query_vector(
    model: InstructorONNX,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
) -> np.ndarray:
    w_r, w_i, w_e = weights

    jd_retrieval = model.encode(
        [[RETRIEVAL_INSTRUCTION, JD_RETRIEVAL_TEXT]], batch_size=1, normalize=False
    ).reshape(-1)
    jd_infra = model.encode(
        [[INFRA_INSTRUCTION, JD_INFRA_TEXT]], batch_size=1, normalize=False
    ).reshape(-1)
    jd_eval = model.encode(
        [[EVAL_INSTRUCTION, JD_EVAL_TEXT]], batch_size=1, normalize=False
    ).reshape(-1)

    jd_retrieval = _normalize_single_block(jd_retrieval) * w_r
    jd_infra = _normalize_single_block(jd_infra) * w_i
    jd_eval = _normalize_single_block(jd_eval) * w_e

    return np.concatenate([jd_retrieval, jd_infra, jd_eval]).astype(np.float32)


def log_encode_plan(batch_size: int, candidate_count: int, passage_workers: int) -> None:
    print("Encode backend: INSTRUCTOR-large ONNX (CUDA)")
    print(f"ONNX batch_size: {batch_size}")
    print(
        f"Candidates: {candidate_count:,} — 3 encode passes "
        f"(~{candidate_count * 3:,} forward batches)"
    )
    print(f"Passage prep workers: {passage_workers} (CPU)")
    print(f"Vector dimension: {VECTOR_DIM}")
