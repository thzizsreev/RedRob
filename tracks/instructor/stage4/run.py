#!/usr/bin/env python3
"""
Stage 4 — Cross-encoder reranking (Phase B, CPU ONNX).

Requires Phase A export first:
    pip install -r models/requirements.txt
    python models/export_cross_encoder.py

Then run after Stage 3:
    python tracks/instructor/stage4/run.py

Inputs:
  STAGE3_INPUT   artifacts/runtime/stage3/stage3_retrieved.parquet
  CONFIG_PATH    config.yaml (stage4: block)

Outputs (under OUTPUT_DIR = artifacts/runtime/stage4/):
  stage4_reranked.parquet
  stage4_reranked.json
  stage4_rank_delta.csv
  stage4_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage4 import print_stage4_summary, run
from tracks.shared.paths import (
    CANDIDATES_JSONL_PATH,
    ROOT_DIR,
    RUNTIME_STAGE3_DIR,
    RUNTIME_STAGE4_DIR,
)

# --- edit before run ---
STAGE3_INPUT = RUNTIME_STAGE3_DIR / "stage3_retrieved.parquet"
OUTPUT_DIR = RUNTIME_STAGE4_DIR
CONFIG_PATH = ROOT_DIR / "config.yaml"
CANDIDATES_PATH = CANDIDATES_JSONL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 cross-encoder reranking.")
    parser.add_argument(
        "--in",
        dest="stage3_path",
        type=Path,
        default=STAGE3_INPUT,
        help="Stage 3 retrieved parquet",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for Stage 4 artifacts",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=CONFIG_PATH,
        help="config.yaml path",
    )
    parser.add_argument(
        "--candidates",
        dest="candidates_path",
        type=Path,
        default=CANDIDATES_PATH,
        help="Candidates JSONL for passage fallback",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = perf_counter()
    result = run(
        stage3_path=args.stage3_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
        candidates_path=args.candidates_path,
    )
    print_stage4_summary(result)
    total = perf_counter() - start_time
    print(f"Stage 4 rerank completed in {total:.2f} seconds")


if __name__ == "__main__":
    main()
