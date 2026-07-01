"""Concurrent candidate processing — one full candidate per thread."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tracks.instructor.stage6.reasoning_builder import (
    merge_precomputed_raw,
    paraphrase_and_reconstruct,
)


def process_candidates_parallel(
    candidates: list[dict[str, Any]],
    raw_cache: dict[str, dict[str, Any]],
    paraphrase_fn: Callable[[str, float], str],
    worker_count: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    def _work(candidate: dict[str, Any]) -> dict[str, Any]:
        cid = str(candidate["candidate_id"])
        raw = merge_precomputed_raw(candidate, raw_cache.get(cid))
        return paraphrase_and_reconstruct(raw, paraphrase_fn)

    if worker_count <= 1 or len(candidates) == 1:
        return [_work(c) for c in candidates]

    results: list[dict[str, Any] | None] = [None] * len(candidates)
    index_by_id = {str(c["candidate_id"]): i for i, c in enumerate(candidates)}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_work, c): c for c in candidates}
        for future in as_completed(futures):
            result = future.result()
            idx = index_by_id[str(result["candidate_id"])]
            results[idx] = result

    return [r for r in results if r is not None]
