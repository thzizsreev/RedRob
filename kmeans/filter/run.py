#!/usr/bin/env python3
"""
K-means Phase C — rank clusters by anchor similarity and filter to floor.

Requires:
  - kmeans/precompute.py (Phase A)
  - kmeans/test/run.py (Phase B) for the same N_CLUSTERS

Edit config below before each run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kmeans.filter.artifacts import (
    load_run_labels,
    print_filter_summary,
    write_filter_artifacts,
)
from kmeans.filter.pipeline import filter_from_labels
from kmeans.io import load_anchor_vector
from kmeans.precompute_artifacts import (
    CANDIDATE_VECTORS_FILENAME,
    require_precompute_artifacts,
)
from tracks.shared.paths import ROOT_DIR

# --- edit before run ---
POOL_TAG = "candidates_full"
PRECOMPUTE_DIR = ROOT_DIR / "kmeans" / "precomputed" / POOL_TAG
N_CLUSTERS = 40
RUN_DIR = PRECOMPUTE_DIR / "runs" / f"k{N_CLUSTERS}"
OUTPUT_DIR = RUN_DIR / "filter"

FLOOR = 100


def run_kmeans_filter(
    precompute_dir: Path,
    run_dir: Path,
    output_dir: Path,
    *,
    n_clusters: int,
    floor: int,
) -> None:
    print("Phase C — K-means rank + filter")
    print(f"Precompute: {precompute_dir}")
    print(f"Run:        {run_dir}")
    print(f"Output:     {output_dir}")

    precompute = require_precompute_artifacts(precompute_dir)
    manifest = precompute.manifest
    candidate_ids = precompute.candidate_ids

    labels = load_run_labels(run_dir, candidate_ids)
    artifacts_path = Path(manifest.artifacts_path)

    print(f"Loading vectors from {precompute_dir / CANDIDATE_VECTORS_FILENAME}...")
    vectors = precompute.vectors
    anchor_vec = load_anchor_vector(artifacts_path)

    result = filter_from_labels(
        candidate_ids,
        vectors,
        labels,
        anchor_vec,
        floor=floor,
    )

    print_filter_summary(result, floor=floor)

    write_filter_artifacts(
        output_dir,
        result,
        candidate_ids=candidate_ids,
        floor=floor,
        n_clusters_k=n_clusters,
        precompute_dir=precompute_dir,
        run_dir=run_dir,
    )
    print(f"\nWrote filter artifacts to {output_dir}")


def main() -> None:
    start_time = perf_counter()
    run_kmeans_filter(
        PRECOMPUTE_DIR,
        RUN_DIR,
        OUTPUT_DIR,
        n_clusters=N_CLUSTERS,
        floor=FLOOR,
    )
    elapsed = perf_counter() - start_time
    print(f"K-means filter completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
