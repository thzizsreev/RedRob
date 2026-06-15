"""Thread-local encoding workers and dynamic worker count for precompute."""

from __future__ import annotations

import os
import threading

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer

from pipeline.candidate import process_one_candidate
from pipeline.config import (
    MAX_PRECOMPUTE_WORKERS,
    MODEL_NAME,
    MODEL_RAM_GB_ESTIMATE,
    PRECOMPUTE_RAM_RESERVE_GB,
    PRECOMPUTE_WORKERS,
)
from pipeline.model_utils import load_sentence_transformer, resolve_device

_thread_local = threading.local()


def choose_worker_count(
    *,
    model_ram_gb: float = MODEL_RAM_GB_ESTIMATE,
    reserve_gb: float = PRECOMPUTE_RAM_RESERVE_GB,
    max_workers_cap: int = MAX_PRECOMPUTE_WORKERS,
    override: int | None = PRECOMPUTE_WORKERS,
) -> int:
    """Pick worker count from CPU cores and available RAM."""
    if override is not None:
        workers = max(1, override)
        print(f"Precompute workers: {workers} (manual override)")
        return workers

    cpu_count = os.cpu_count() or 4
    cpu_workers = max(1, cpu_count - 1)

    available_gb = psutil.virtual_memory().available / (1024**3)
    ram_workers = max(1, int((available_gb - reserve_gb) // model_ram_gb))

    workers = min(cpu_workers, ram_workers, max_workers_cap)
    workers = max(1, workers)

    print(
        f"Precompute workers: {workers} "
        f"(cpus={cpu_count}, available_ram={available_gb:.1f}GB, "
        f"cpu_limit={cpu_workers}, ram_limit={ram_workers})"
    )
    return workers


def get_thread_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load one SentenceTransformer per worker thread (CPU only)."""
    if not hasattr(_thread_local, "model"):
        torch.set_num_threads(1)
        _thread_local.model = load_sentence_transformer(model_name, device="cpu")
    return _thread_local.model


def encode_candidate_task(
    args: tuple[int, dict, dict[str, np.ndarray], dict[str, float], str],
) -> tuple[int, str, np.ndarray]:
    """Encode a single candidate record (runs inside a worker thread)."""
    idx, record, anchors, thresholds, model_name = args
    model = get_thread_model(model_name)
    candidate_id, vec = process_one_candidate(record, model, anchors, thresholds)
    return idx, candidate_id, vec


def resolve_workers(workers: int | None) -> int:
    if resolve_device() == "cuda":
        print("GPU encode: using 1 worker")
        return 1
    if workers is not None:
        return max(1, workers)
    return choose_worker_count()
