#!/usr/bin/env python3
"""
Stage 1 cluster-based filtering test.

Run from project root (after precompute on the same sample):
    python tests/run_filtering_test.py

Edit the TRACK block below before each run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.config import STAGE1_FLOOR
from tracks.instructor.filtering.pipeline import run_stage1_filtering
from tracks.instructor.io import load_jd_query_vector, load_vectors_from_artifacts
from tracks.instructor.config import INDEX_FILENAME as INSTRUCTOR_INDEX
from tracks.instructor.config import VECTOR_DIM as INSTRUCTOR_VECTOR_DIM
from tracks.shared.paths import ARTIFACTS_DIR, ROOT_DIR

# --- edit before run ---
TRACK = "instructor"  # instructor only for Stage 1
SAMPLE_TAG = "sample1k"

RANDOM_SEED = 42

if TRACK == "instructor":
    ARTIFACTS_PATH = ARTIFACTS_DIR / SAMPLE_TAG
    OUTPUT_DIR = ROOT_DIR / "test_output" / "filtering" / "instructor" / SAMPLE_TAG
    INDEX_FILENAME = INSTRUCTOR_INDEX
    VECTOR_DIM = INSTRUCTOR_VECTOR_DIM
else:
    raise ValueError(f"Stage 1 filtering supports instructor track only, got {TRACK!r}")


def main() -> None:
    print(f"Artifacts: {ARTIFACTS_PATH}")
    print(f"Output:    {OUTPUT_DIR}")

    candidate_ids, vectors = load_vectors_from_artifacts(
        ARTIFACTS_PATH,
        index_filename=INDEX_FILENAME,
        vector_dim=VECTOR_DIM,
    )
    anchor_vec = load_jd_query_vector(ARTIFACTS_PATH)
    print(f"Loaded {len(candidate_ids):,} candidates")

    result = run_stage1_filtering(
        candidate_ids,
        vectors,
        anchor_vec,
        random_seed=RANDOM_SEED,
    )

    print(f"\n--- Stage 1 summary ---")
    print(f"Clusters:     {result.n_clusters}")
    print(f"Noise:        {result.noise_count} ({result.noise_ratio:.1%})")
    print(f"Filtered set: {len(result.filtered_ids)} (floor={STAGE1_FLOOR})")

    print("\n--- Ranked clusters (label, median_sim, size) ---")
    for label, median_sim, size in result.ranked_clusters:
        print(f"  {label:4d}  median={median_sim:.4f}  size={size}")

    if len(result.filtered_ids) < STAGE1_FLOOR:
        print(
            f"\nWARNING: filtered set ({len(result.filtered_ids)}) "
            f"is below floor ({STAGE1_FLOOR})"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "filtered_ids.json", "w", encoding="utf-8") as f:
        json.dump(sorted(result.filtered_ids), f, indent=2)

    rankings = [
        {"label": label, "median_similarity": median_sim, "size": size}
        for label, median_sim, size in result.ranked_clusters
    ]
    with open(OUTPUT_DIR / "cluster_rankings.json", "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)

    summary = {
        "n_candidates": len(candidate_ids),
        "n_clusters": result.n_clusters,
        "noise_count": result.noise_count,
        "noise_ratio": result.noise_ratio,
        "filtered_count": len(result.filtered_ids),
        "floor": STAGE1_FLOOR,
        "random_seed": RANDOM_SEED,
    }
    with open(OUTPUT_DIR / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
