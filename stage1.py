#!/usr/bin/env python3
"""
Stage 1 — cluster-based filtering (Track A).

Runs UMAP + HDBSCAN clustering, ranks clusters by median JD anchor similarity,
and selects candidates until floor is met. Edit ARTIFACTS_PATH and OUTPUT_DIR below.
"""

from __future__ import annotations

from tracks.instructor.stage1 import run_stage1_from_artifacts
from tracks.shared.paths import ROOT_DIR

# --- edit before run ---
ARTIFACTS_PATH = ROOT_DIR / "artifacts" / "sample1k"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "sample1k" / "stage1"


def main() -> None:
    run_stage1_from_artifacts(ARTIFACTS_PATH, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
