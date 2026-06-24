#!/usr/bin/env python3
"""
Stage 3 — Multi-Query Hybrid Retrieval.

Requires Stage 2 output and Stage 0 precomputed artifacts. Edit paths below before running.

    python tracks/instructor/stage3/run.py

Inputs:
  STAGE0_PATH            artifacts/runtime/stage0/ (FAISS index, vectors, BM25)
  STAGE2_INPUT           artifacts/runtime/stage2/stage2_gated.parquet
  CONFIG_PATH            config.yaml (stage3: block)

Outputs (under OUTPUT_DIR = artifacts/runtime/stage3/):
  stage3_retrieved.parquet
  stage3_retrieved.json
  stage3_score_distribution.csv
  stage3_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage3 import print_stage3_summary, run
from tracks.shared.paths import (
    ROOT_DIR,
    RUNTIME_STAGE0_DIR,
    RUNTIME_STAGE2_DIR,
    RUNTIME_STAGE3_DIR,
)

# --- edit before run ---
STAGE0_PATH = RUNTIME_STAGE0_DIR
STAGE2_INPUT = RUNTIME_STAGE2_DIR / "stage2_gated.parquet"
OUTPUT_DIR = RUNTIME_STAGE3_DIR
CONFIG_PATH = ROOT_DIR / "config.yaml"

# STAGE2_INPUT = ROOT_DIR / "kmeans" / "outputs" / "candidates_full" / "stage2" / "stage2_gated.parquet"
# OUTPUT_DIR = ROOT_DIR / "kmeans" / "outputs" / "candidates_full" / "stage3"
# STAGE0_PATH = ROOT_DIR / "kmeans" / "precomputed" / "candidates_full" / "runs" / "k40"


def main() -> None:
    start_time = perf_counter()
    result = run(
        stage2_path=STAGE2_INPUT,
        artifacts_path=STAGE0_PATH,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
    )
    print_stage3_summary(result)
    total = perf_counter() - start_time
    print(f"Stage 3 retrieval completed in {total:.2f} seconds")


if __name__ == "__main__":
    main()
