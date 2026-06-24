#!/usr/bin/env python3
"""
Honeypot LLM research pipeline — main runner (not used by rank.py).

Run from project root:
    python honeypot/run.py

Edit the config block below before each run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from honeypot.config import (
    DEFAULT_INITIAL_BACKOFF_SEC,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PER_STRATUM,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REQUESTS_PER_MINUTE,
    PipelineConfig,
    load_env,
    resolve_openai_model,
)
from honeypot.pipeline import run_pipeline
from tracks.shared.paths import CANDIDATES_JSONL_PATH, ROOT_DIR, SAMPLE1K_PATH

# --- edit before run ---
FILTERED_IDS_PATH = ROOT_DIR / "data" / "filter_kmeans" / "filtered_ids.json"
CANDIDATES_PATH = CANDIDATES_JSONL_PATH
OUTPUT_DIR = ROOT_DIR / "honeypot" / "output" / "filter_kmeans"

# Stratified sampling (set FILTERED_IDS_PATH = None to use these instead)
# FILTERED_IDS_PATH = None
# CANDIDATES_PATH = SAMPLE1K_PATH
# OUTPUT_DIR = ROOT_DIR / "honeypot" / "output"

PER_STRATUM = DEFAULT_PER_STRATUM
RANDOM_SEED = DEFAULT_RANDOM_SEED
MAX_WORKERS = DEFAULT_MAX_WORKERS
REQUESTS_PER_MINUTE = DEFAULT_REQUESTS_PER_MINUTE
PASS_MODE = "all"  # "1", "2", or "all"
SAMPLE_ONLY = False
FORCE = False
MANIFEST_PATH: Path | None = None
VERBOSE = True


def main() -> None:
    load_env()
    config = PipelineConfig(
        candidates_path=CANDIDATES_PATH,
        output_dir=OUTPUT_DIR,
        openai_model=resolve_openai_model(),
        max_workers=MAX_WORKERS,
        requests_per_minute=REQUESTS_PER_MINUTE,
        max_retries=DEFAULT_MAX_RETRIES,
        initial_backoff_sec=DEFAULT_INITIAL_BACKOFF_SEC,
        per_stratum=PER_STRATUM,
        random_seed=RANDOM_SEED,
        force=FORCE,
        sample_only=SAMPLE_ONLY,
        pass_mode=PASS_MODE,
        manifest_path=MANIFEST_PATH,
        filtered_ids_path=FILTERED_IDS_PATH,
        verbose=VERBOSE,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
