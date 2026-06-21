#!/usr/bin/env python3
"""
Stage 1 cluster-based filtering test (two-phase).

Mirrors the production entry scripts:
    python stage1_cluster.py   # Phase A
    python stage1.py           # Phase B

Run from project root (after precompute on the same sample):
    python tests/run_filtering_test.py

Edit the TRACK block below before each run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.config import STAGE1_FLOOR, STAGE1_DIRNAME
from tracks.instructor.stage1 import (
    precompute_stage1_clustering,
    print_stage1_summary,
    run_stage1_filter,
)
from tracks.instructor.config import INDEX_FILENAME as INSTRUCTOR_INDEX
from tracks.instructor.config import VECTOR_DIM as INSTRUCTOR_VECTOR_DIM
from tracks.shared.paths import ARTIFACTS_DIR, ROOT_DIR

# --- edit before run ---
TRACK = "instructor"  # instructor only for Stage 1
SAMPLE_TAG = "sample1k"

RANDOM_SEED = 42
OVERWRITE_CLUSTER = False

if TRACK == "instructor":
    ARTIFACTS_PATH = ARTIFACTS_DIR / SAMPLE_TAG
    STAGE1_PATH = ARTIFACTS_PATH / STAGE1_DIRNAME
    OUTPUT_DIR = ROOT_DIR / "test_output" / "filtering" / "instructor" / SAMPLE_TAG
    INDEX_FILENAME = INSTRUCTOR_INDEX
    VECTOR_DIM = INSTRUCTOR_VECTOR_DIM
else:
    raise ValueError(f"Stage 1 filtering supports instructor track only, got {TRACK!r}")


def main() -> None:
    print(f"Artifacts: {ARTIFACTS_PATH}")
    print(f"Stage1:    {STAGE1_PATH}")
    print(f"Output:    {OUTPUT_DIR}")

    precompute_stage1_clustering(
        ARTIFACTS_PATH,
        stage1_path=STAGE1_PATH,
        random_seed=RANDOM_SEED,
        index_filename=INDEX_FILENAME,
        vector_dim=VECTOR_DIM,
        overwrite=OVERWRITE_CLUSTER,
    )

    run_result = run_stage1_filter(
        ARTIFACTS_PATH,
        stage1_path=STAGE1_PATH,
        output_dir=OUTPUT_DIR,
        random_seed=RANDOM_SEED,
        index_filename=INDEX_FILENAME,
        print_summary=False,
    )
    result = run_result.result

    print(f"\n--- Stage 1 summary ---")
    print_stage1_summary(result, floor=STAGE1_FLOOR)

    print(f"\nWrote filter artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
