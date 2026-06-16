"""INSTRUCTOR-large batched encoding for Track A (precompute only — never import from rank.py)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import faiss
import numpy as np
import torch
from huggingface_hub import snapshot_download
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer

from pipeline.config import (
    BLOCK_DIM,
    ENCODE_DEVICE,
    EVAL_INSTRUCTION,
    INFRA_INSTRUCTION,
    INSTRUCTOR_BATCH_SIZE,
    INSTRUCTOR_BATCH_SIZE_CPU,
    INSTRUCTOR_MODEL,
    JD_EVAL_TEXT,
    JD_INFRA_TEXT,
    JD_RETRIEVAL_TEXT,
    QUERY_WEIGHTS,
    RETRIEVAL_INSTRUCTION,
    VECTOR_DIM,
)

if TYPE_CHECKING:
    pass


def _patch_instructor_compatibility() -> None:
    """Adapt INSTRUCTOR 1.0.1 to sentence-transformers 2.7+ hub loading."""
    orig_load = INSTRUCTOR._load_sbert_model

    def _load_sbert_model_compat(self, model_path, *args, **kwargs):
        modules_json = os.path.join(model_path, "modules.json")
        if not os.path.exists(modules_json):
            model_path = snapshot_download(
                model_path,
                cache_dir=kwargs.get("cache_folder"),
                revision=kwargs.get("revision"),
                token=kwargs.get("token"),
            )
        return orig_load(self, model_path)

    INSTRUCTOR._load_sbert_model = _load_sbert_model_compat  # type: ignore[method-assign]


def resolve_device(preference: str | None = None) -> str:
    """Resolve encode device for precompute: auto uses cuda when available."""
    preference = (preference or ENCODE_DEVICE).lower()
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference in ("cuda", "cpu"):
        if preference == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return preference
    raise ValueError(f"Unsupported ENCODE_DEVICE: {preference}")


def batch_size_for_device(device: str, override: int | None = None) -> int:
    if override is not None:
        return max(1, override)
    return INSTRUCTOR_BATCH_SIZE if device == "cuda" else INSTRUCTOR_BATCH_SIZE_CPU


def load_tokenizer(model_name: str = INSTRUCTOR_MODEL):
    return AutoTokenizer.from_pretrained(model_name)


def load_instructor(
    model_name: str = INSTRUCTOR_MODEL,
    device: str | None = None,
) -> INSTRUCTOR:
    resolved = device or resolve_device()
    print(f"Loading INSTRUCTOR model: {model_name}")
    _patch_instructor_compatibility()
    model = INSTRUCTOR(model_name)
    if resolved == "cuda":
        model = model.cuda()
    model.eval()
    print(f"  device: {resolved}")
    if resolved == "cuda" and torch.cuda.is_available():
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    return model


def unload_instructor(model: INSTRUCTOR | None) -> None:
    if model is None:
        return
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_at_batch_size(
    model: INSTRUCTOR,
    pairs: list[list[str]],
    batch_size: int,
) -> np.ndarray:
    with torch.no_grad():
        vecs = model.encode(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True,
        )
    return np.asarray(vecs, dtype=np.float32)


def encode_instruction_pairs(
    model: INSTRUCTOR,
    instruction: str,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    pairs = [[instruction, t] for t in texts]
    fallbacks = [batch_size]
    for size in (INSTRUCTOR_BATCH_SIZE, 16, INSTRUCTOR_BATCH_SIZE_CPU):
        if size not in fallbacks:
            fallbacks.append(size)

    last_error: Exception | None = None
    for size in fallbacks:
        try:
            if size != batch_size:
                print(f"  Retrying encode with batch_size={size}")
            return _encode_at_batch_size(model, pairs, size)
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    if last_error is not None:
        raise RuntimeError("INSTRUCTOR encode OOM at minimum batch size") from last_error
    return _encode_at_batch_size(model, pairs, batch_size)


def normalize_by_block(vecs: np.ndarray, block_size: int = BLOCK_DIM) -> np.ndarray:
    """L2-normalize each 768-d block independently, then concatenate."""
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
    model: INSTRUCTOR,
    passages: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    """Three batched INSTRUCTOR passes; returns [N, 2304] per-block-normalized vectors."""
    retrieval_vecs = encode_instruction_pairs(
        model, RETRIEVAL_INSTRUCTION, passages, batch_size
    )
    infra_vecs = encode_instruction_pairs(
        model, INFRA_INSTRUCTION, passages, batch_size
    )
    eval_vecs = encode_instruction_pairs(
        model, EVAL_INSTRUCTION, passages, batch_size
    )
    combined = np.concatenate([retrieval_vecs, infra_vecs, eval_vecs], axis=1)
    return normalize_by_block(combined)


def build_jd_query_vector(
    model: INSTRUCTOR,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
) -> np.ndarray:
    """Encode JD subspace texts with matching instructions and apply block weights."""
    w_r, w_i, w_e = weights

    with torch.no_grad():
        jd_retrieval = np.asarray(
            model.encode([[RETRIEVAL_INSTRUCTION, JD_RETRIEVAL_TEXT]]),
            dtype=np.float32,
        ).reshape(-1)
        jd_infra = np.asarray(
            model.encode([[INFRA_INSTRUCTION, JD_INFRA_TEXT]]),
            dtype=np.float32,
        ).reshape(-1)
        jd_eval = np.asarray(
            model.encode([[EVAL_INSTRUCTION, JD_EVAL_TEXT]]),
            dtype=np.float32,
        ).reshape(-1)

    jd_retrieval = _normalize_single_block(jd_retrieval) * w_r
    jd_infra = _normalize_single_block(jd_infra) * w_i
    jd_eval = _normalize_single_block(jd_eval) * w_e

    return np.concatenate([jd_retrieval, jd_infra, jd_eval]).astype(np.float32)


def log_encode_plan(device: str, batch_size: int, candidate_count: int, passage_workers: int) -> None:
    gpu_name = ""
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = f" ({torch.cuda.get_device_name(0)})"
    print(f"Device: {device}{gpu_name}")
    print(f"INSTRUCTOR batch_size: {batch_size}")
    print(
        f"Candidates: {candidate_count:,} — 3 encode passes "
        f"(~{candidate_count * 3:,} forward batches)"
    )
    print(f"Passage prep workers: {passage_workers} (CPU)")
    print(f"Vector dimension: {VECTOR_DIM}")
