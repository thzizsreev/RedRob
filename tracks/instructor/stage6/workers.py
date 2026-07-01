"""Dynamic worker count for Stage 6 paraphrase pool."""

from __future__ import annotations

import os


def _mem_cap_from_available(
    estimated_session_mb: int,
    memory_reserve_ratio: float,
) -> int | None:
    try:
        import psutil

        available_mb = psutil.virtual_memory().available // (1024 * 1024)
        usable = int(available_mb * (1.0 - memory_reserve_ratio))
        return max(1, usable // max(estimated_session_mb, 1))
    except ImportError:
        return None


def resolve_worker_count(
    n_candidates: int,
    *,
    cpu_count: int | None = None,
    ort_intra_op_threads: int = 1,
    estimated_session_mb: int = 700,
    memory_reserve_ratio: float = 0.25,
    max_workers: int | None = None,
) -> tuple[int, dict]:
    cpus = max(1, (cpu_count or os.cpu_count() or 4))
    cpu_cap = max(1, (cpus - 1) // max(ort_intra_op_threads, 1))
    mem_cap = _mem_cap_from_available(estimated_session_mb, memory_reserve_ratio)

    caps = {"cpu_cap": cpu_cap, "mem_cap": mem_cap, "candidate_cap": n_candidates}
    worker_count = min(n_candidates, cpu_cap)
    if mem_cap is not None:
        worker_count = min(worker_count, mem_cap)
    if max_workers is not None:
        worker_count = min(worker_count, max_workers)
    worker_count = max(1, worker_count)

    rationale = {
        "cpus": cpus,
        "ort_intra_op_threads": ort_intra_op_threads,
        "cpu_cap": cpu_cap,
        "mem_cap": mem_cap,
        "max_workers_override": max_workers,
        "chosen": worker_count,
    }
    return worker_count, rationale
