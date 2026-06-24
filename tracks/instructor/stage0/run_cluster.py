#!/usr/bin/env python3
"""
Stage 0 — cluster precompute (UMAP + HDBSCAN).

Run once per candidate pool after Stage 0 vector precompute (run.py).

    python tracks/instructor/stage0/run_cluster.py

Inputs:
  STAGE0_PATH   artifacts/runtime/stage0/  (FAISS index + vectors)

Outputs (under STAGE1_PATH = artifacts/runtime/stage1/):
  candidate_vectors.npy
  cluster_labels.npy
  umap_reduced_12d.npy
  cluster_manifest.json

Then run tracks/instructor/stage1/run_filter.py (Phase B).
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.core.config import STAGE1_RANDOM_SEED
from tracks.instructor.stage0.cluster_precompute import run_cluster_precompute
from tracks.shared.paths import RUNTIME_STAGE0_DIR, RUNTIME_STAGE1_DIR

# --- edit before run ---
STAGE0_PATH = RUNTIME_STAGE0_DIR
STAGE1_PATH = RUNTIME_STAGE1_DIR
OVERWRITE = False
RANDOM_SEED = STAGE1_RANDOM_SEED


def main() -> None:
    start_time = perf_counter()
    run_cluster_precompute(
        STAGE0_PATH,
        STAGE1_PATH,
        random_seed=RANDOM_SEED,
        overwrite=OVERWRITE,
    )
    elapsed = perf_counter() - start_time
    print(f"Stage 0 cluster precompute completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
