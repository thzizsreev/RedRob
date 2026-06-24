#!/usr/bin/env python3
"""
Stage 0 — INSTRUCTOR-large ONNX block-weighted vector precomputation.

GPU via ONNX Runtime (CUDA) only. Edit CANDIDATES_PATH and OUTPUT_DIR below before running.

    python tracks/instructor/stage0/run.py

Outputs (under OUTPUT_DIR = artifacts/runtime/stage0/):
  candidate_index.faiss  (IndexFlatIP, 2304-d)
  id_map.json
  jd_query_vec.npy
  candidate_vectors.npy
  bm25_index.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.onnx_embedder import load_embedder, unload_embedder
from tracks.instructor.stage0.precompute import run_precompute
from tracks.shared.paths import (
    ARTIFACTS_DIR,
    CANDIDATES_JSONL_PATH,
    ROOT_DIR,
    RUNTIME_STAGE0_DIR,
)

CONFIG_PATH = ROOT_DIR / "config.yaml"

# --- edit before run ---
CANDIDATES_PATH = CANDIDATES_JSONL_PATH
OUTPUT_DIR = RUNTIME_STAGE0_DIR


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
        run_precompute(CANDIDATES_PATH, model, OUTPUT_DIR, config_path=CONFIG_PATH)
    finally:
        unload_embedder(model)
    elapsed = perf_counter() - started
    print(f"Precompute complete in {_format_duration(elapsed)}.")


def main_full() -> None:
    print(f"Using candidates file: {CANDIDATES_JSONL_PATH}")
    model = load_embedder()
    started = perf_counter()
    try:
        run_precompute(CANDIDATES_JSONL_PATH, model, ARTIFACTS_DIR, config_path=CONFIG_PATH)
    finally:
        unload_embedder(model)
    elapsed = perf_counter() - started
    print(f"Precompute complete in {_format_duration(elapsed)}.")


if __name__ == "__main__":
    main()
