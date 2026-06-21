#!/usr/bin/env python3
"""
Offline Phase 1 — INSTRUCTOR-large ONNX block-weighted vector precomputation.

GPU via ONNX Runtime (CUDA) only. Edit CANDIDATES_PATH and OUTPUT_DIR below before running.
Outputs (under OUTPUT_DIR):
  candidate_index.faiss  (IndexFlatIP, 2304-d)
  id_map.json
  jd_query_vec.npy
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from tracks.instructor.index import build_vector_index, build_vector_index_from_records
from tracks.instructor.onnx_embedder import load_embedder, unload_embedder
from tracks.shared.paths import (
    ARTIFACTS_DIR,
    CANDIDATES_JSONL_PATH,
    ROOT_DIR,
    SAMPLE1K_PATH,
    SAMPLE10K_PATH,
    SAMPLE5K_PATH,
    SAMPLE20K_PATH,
)

# --- edit before run ---
CANDIDATES_PATH = CANDIDATES_JSONL_PATH
OUTPUT_DIR = ROOT_DIR / "artifacts" / "candidates_full"


def load_candidates_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def run_precompute(
    candidates_path: Path,
    model,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to {output_dir}")

    if candidates_path.suffix == ".json":
        candidates = load_candidates_json(candidates_path)
        print(f"Processing {len(candidates)} candidates from {candidates_path}...")
        build_vector_index_from_records(candidates, model, output_dir)
    else:
        build_vector_index(candidates_path, model, output_dir)

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
    print(f"Using candidates file: {CANDIDATES_PATH}")
    model = load_embedder()
    started = perf_counter()
    try:
        run_precompute(CANDIDATES_PATH, model, OUTPUT_DIR)
    finally:
        unload_embedder(model)
    elapsed = perf_counter() - started
    print(f"Precompute complete in {_format_duration(elapsed)}.")


def main_full() -> None:
    print(f"Using candidates file: {CANDIDATES_JSONL_PATH}")
    model = load_embedder()
    started = perf_counter()
    try:
        run_precompute(CANDIDATES_JSONL_PATH, model, ARTIFACTS_DIR)
    finally:
        unload_embedder(model)
    elapsed = perf_counter() - started
    print(f"Precompute complete in {_format_duration(elapsed)}.")


if __name__ == "__main__":
    main()
