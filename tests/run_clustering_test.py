#!/usr/bin/env python3
"""
Semantic richness clustering test — Stages 1–6 and basic Stage 8.

Run from project root (after precompute on the same sample):
    python tests/run_clustering_test.py

Edit the TRACK block below before each run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.clustering.pipeline import run_clustering_pipeline
from tracks.instructor.core.config import INDEX_FILENAME as INSTRUCTOR_INDEX
from tracks.instructor.core.config import VECTOR_DIM as INSTRUCTOR_VECTOR_DIM
from tracks.naive.config import (
    NAIVE_ARTIFACTS_DIR,
    NAIVE_INDEX_FILENAME,
    NAIVE_VECTOR_DIM,
)
from tracks.shared.paths import ARTIFACTS_DIR, ROOT_DIR, SAMPLE1K_PATH, CANDIDATES_JSONL_PATH

# --- edit before run ---
TRACK = "instructor"  # "naive" | "instructor"
SAMPLE_TAG = "candidates_full"

CANDIDATES_PATH = CANDIDATES_JSONL_PATH
SAMPLE_SIZE = 100000
RANDOM_SEED = 42
LANDMARK_CANDIDATE_IDS: list[str] = []
ENABLE_ID_SEARCH = True

if TRACK == "naive":
    # Naive precompute writes to tracks/naive/artifacts/ (no per-sample subdir)
    ARTIFACTS_PATH = NAIVE_ARTIFACTS_DIR
    OUTPUT_DIR = ROOT_DIR / "test_output" / "clustering" / "naive" / SAMPLE_TAG
    INDEX_FILENAME = NAIVE_INDEX_FILENAME
    VECTOR_DIM = NAIVE_VECTOR_DIM
elif TRACK == "instructor":
    # Instructor precompute writes to artifacts/<sample>/ (e.g. artifacts/sample5k/)
    ARTIFACTS_PATH = ARTIFACTS_DIR / SAMPLE_TAG
    OUTPUT_DIR = ROOT_DIR / "test_output" / "clustering" / "instructor" / SAMPLE_TAG
    INDEX_FILENAME = INSTRUCTOR_INDEX
    VECTOR_DIM = INSTRUCTOR_VECTOR_DIM
else:
    raise ValueError(f"Unknown TRACK: {TRACK!r}")


def main() -> None:
    run_clustering_pipeline(
        candidates_path=CANDIDATES_PATH,
        artifacts_path=ARTIFACTS_PATH,
        output_dir=OUTPUT_DIR,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED,
        landmark_ids=LANDMARK_CANDIDATE_IDS,
        enable_id_search=ENABLE_ID_SEARCH,
        index_filename=INDEX_FILENAME,
        vector_dim=VECTOR_DIM,
    )


if __name__ == "__main__":
    main()
