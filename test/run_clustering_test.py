#!/usr/bin/env python3
"""
Semantic richness clustering test — Stages 1–6 and basic Stage 8.

Run from project root (after precompute.py on the same sample):
    python test/run_clustering_test.py

Edit CANDIDATES_PATH, ARTIFACTS_PATH, and OUTPUT_DIR below before each run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(TEST_DIR))

from pipeline.config import ARTIFACTS_DIR, SAMPLE1K_PATH
from clustering.pipeline import run_clustering_pipeline

# --- edit before run ---
CANDIDATES_PATH = SAMPLE1K_PATH
ARTIFACTS_PATH = ARTIFACTS_DIR / "sample1k"
OUTPUT_DIR = ROOT_DIR / "test_output" / "clustering" / "sample1k"

SAMPLE_SIZE = 3000
RANDOM_SEED = 42
LANDMARK_CANDIDATE_IDS: list[str] = []
ENABLE_ID_SEARCH = True


def main() -> None:
    run_clustering_pipeline(
        candidates_path=CANDIDATES_PATH,
        artifacts_path=ARTIFACTS_PATH,
        output_dir=OUTPUT_DIR,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED,
        landmark_ids=LANDMARK_CANDIDATE_IDS,
        enable_id_search=ENABLE_ID_SEARCH,
    )


if __name__ == "__main__":
    main()
