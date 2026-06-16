#!/usr/bin/env python3
"""
Offline Phase 1 — INSTRUCTOR-large block-weighted vector precomputation.

GPU/torch allowed here only. Outputs:
  artifacts/candidate_index.faiss  (IndexFlatIP, 2304-d)
  artifacts/id_map.json
  artifacts/jd_query_vec.npy
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from pipeline.config import (
    ARTIFACTS_DIR,
    CANDIDATES_JSONL_PATH,
    SAMPLE_CANDIDATES_PATH,
)
from pipeline.index import build_vector_index, build_vector_index_from_records
from pipeline.instructor_encode import load_instructor, resolve_device, unload_instructor


def load_candidates_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def run_precompute(
    candidates_path: Path,
    model,
    *,
    device: str,
    output_dir: Path = ARTIFACTS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if candidates_path.suffix == ".json":
        candidates = load_candidates_json(candidates_path)
        print(f"Processing {len(candidates)} candidates from {candidates_path}...")
        build_vector_index_from_records(candidates, model, output_dir, device=device)
    else:
        build_vector_index(candidates_path, model, output_dir, device=device)

    print("Precompute complete.")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def main() -> None:
    print(f"Using candidates file: {SAMPLE_CANDIDATES_PATH}")
    device = resolve_device()
    model = load_instructor(device=device)
    started = perf_counter()
    try:
        run_precompute(SAMPLE_CANDIDATES_PATH, model, device=device)
    finally:
        unload_instructor(model)
    elapsed = perf_counter() - started
    print(f"Precompute complete in {_format_duration(elapsed)}.")


def main_full() -> None:
    print(f"Using candidates file: {CANDIDATES_JSONL_PATH}")
    device = resolve_device()
    model = load_instructor(device=device)
    started = perf_counter()
    try:
        run_precompute(CANDIDATES_JSONL_PATH, model, device=device)
    finally:
        unload_instructor(model)
    elapsed = perf_counter() - started
    print(f"Precompute complete in {_format_duration(elapsed)}.")


if __name__ == "__main__":
    main()
