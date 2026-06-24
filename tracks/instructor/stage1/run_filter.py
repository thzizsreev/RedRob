#!/usr/bin/env python3
"""
Stage 1 Phase B — rank clusters and filter to floor.

Requires Phase A artifacts from run_cluster.py. Edit ARTIFACTS_PATH below.

    python tracks/instructor/stage1/run_filter.py

Outputs (under OUTPUT_DIR = artifacts/runtime/stage1/):
  filtered_ids.json
  filtered_metadata.json
  cluster_rankings.json
  stage1_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.config import STAGE1_RANDOM_SEED
from tracks.instructor.stage1 import run_stage1_filter
from tracks.shared.paths import ROOT_DIR, RUNTIME_STAGE0_DIR, RUNTIME_STAGE1_DIR

# --- edit before run ---
STAGE0_PATH = RUNTIME_STAGE0_DIR
STAGE1_PATH = RUNTIME_STAGE1_DIR
OUTPUT_DIR = RUNTIME_STAGE1_DIR
RANDOM_SEED = STAGE1_RANDOM_SEED


def main() -> None:
    start_time = perf_counter()
    run_stage1_filter(
        STAGE0_PATH,
        stage1_path=STAGE1_PATH,
        output_dir=OUTPUT_DIR,
        random_seed=RANDOM_SEED,
    )
    elapsed = perf_counter() - start_time
    print(f"Stage 1 filter completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
