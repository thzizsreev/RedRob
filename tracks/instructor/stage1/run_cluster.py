#!/usr/bin/env python3
"""
Stage 1 Phase A — cluster precompute (UMAP + HDBSCAN).

Run once per candidate pool after Stage 0. Edit ARTIFACTS_PATH below.

    python tracks/instructor/stage1/run_cluster.py

Outputs (under STAGE1_PATH = artifacts/runtime/stage1/):
  candidate_vectors.npy
  cluster_labels.npy
  umap_reduced_12d.npy
  cluster_manifest.json

Then run run_filter.py (Phase B) to rank clusters and write filtered JSON artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.config import STAGE1_RANDOM_SEED
from tracks.instructor.stage1 import precompute_stage1_clustering
from tracks.shared.paths import ROOT_DIR, RUNTIME_STAGE0_DIR, RUNTIME_STAGE1_DIR

# --- edit before run ---
STAGE0_PATH = RUNTIME_STAGE0_DIR
STAGE1_PATH = RUNTIME_STAGE1_DIR
OVERWRITE = False
RANDOM_SEED = STAGE1_RANDOM_SEED


def main() -> None:
    start_time = perf_counter()
    precompute_stage1_clustering(
        STAGE0_PATH,
        stage1_path=STAGE1_PATH,
        random_seed=RANDOM_SEED,
        overwrite=OVERWRITE,
    )
    elapsed = perf_counter() - start_time
    print(f"Stage 1 cluster precompute completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
