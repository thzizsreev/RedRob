#!/usr/bin/env python3
"""
Stage 1 Phase A — cluster precompute (Track A).

Run once per candidate pool after precompute.py. Edit ARTIFACTS_PATH below before running.
Outputs (under STAGE1_PATH):
  candidate_vectors.npy
  cluster_labels.npy
  umap_reduced_12d.npy
  cluster_manifest.json

Then run stage1.py (Phase B) to rank clusters and write filtered JSON artifacts.
"""

from __future__ import annotations

from time import perf_counter

from tracks.instructor.config import STAGE1_RANDOM_SEED
from tracks.instructor.stage1 import precompute_stage1_clustering
from tracks.shared.paths import ROOT_DIR

# --- edit before run ---
ARTIFACTS_PATH = ROOT_DIR / "artifacts" / "candidates_full"
STAGE1_PATH = ARTIFACTS_PATH / "stage1"
OVERWRITE = False
RANDOM_SEED = STAGE1_RANDOM_SEED


def main() -> None:
    start_time = perf_counter()
    precompute_stage1_clustering(
        ARTIFACTS_PATH,
        stage1_path=STAGE1_PATH,
        random_seed=RANDOM_SEED,
        overwrite=OVERWRITE,
    )
    elapsed = perf_counter() - start_time
    print(f"Stage 1 cluster precompute completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
