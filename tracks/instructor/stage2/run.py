#!/usr/bin/env python3
"""
Stage 2 — Hard Tabular Gate.

Requires Stage 1 filter JSON (HDBSCAN or K-means). Edit paths below before running.

    python tracks/instructor/stage2/run.py

Outputs (under OUTPUT_DIR = artifacts/runtime/stage2/):
  stage2_gated.parquet       # downstream processing
  stage2_gated.json          # human-readable survivor table
  stage2_filtered_ids.json   # passed candidate IDs only
  stage2_honeypot_log.csv
  stage2_removed_log.csv
  stage2_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage2 import print_stage2_summary, run
from tracks.shared.paths import (
    ROOT_DIR,
    RUNTIME_STAGE0_DIR,
    RUNTIME_STAGE1_DIR,
    RUNTIME_STAGE2_DIR,
)

# --- edit before run ---
STAGE0_PATH = RUNTIME_STAGE0_DIR
CANDIDATES_PATH = ROOT_DIR / "data" / "candidates.jsonl"
CONFIG_PATH = ROOT_DIR / "config.yaml"

STAGE1_PATH = RUNTIME_STAGE1_DIR
OUTPUT_DIR = RUNTIME_STAGE2_DIR

# STAGE1_PATH = ROOT_DIR / "kmeans" / "precomputed" / "candidates_full" / "runs" / "k40" / "filter"
# OUTPUT_DIR = ROOT_DIR / "kmeans" / "outputs" / "candidates_full" / "stage2"
# STAGE0_PATH = ROOT_DIR / "kmeans" / "precomputed" / "candidates_full" / "runs" / "k40"
def main() -> None:
    start_time = perf_counter()
    result = run(
        stage1_path=STAGE1_PATH,
        artifacts_path=STAGE0_PATH,
        candidates_path=CANDIDATES_PATH,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
    )
    print_stage2_summary(result)
    total = perf_counter() - start_time
    print(f"Stage 2 gate completed in {total:.2f} seconds")


if __name__ == "__main__":
    main()
