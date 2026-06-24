#!/usr/bin/env python3
"""
Stage 5 — Deterministic composite scorer (7-layer formula).

Run after Stage 4:
    python tracks/instructor/stage5/run.py

Inputs:
  STAGE4_INPUT   artifacts/runtime/stage4/stage4_reranked.parquet
  CONFIG_PATH    config.yaml (stage5: block)

Outputs (under OUTPUT_DIR = artifacts/runtime/stage5/):
  {team_id}.csv
  stage5_scored.parquet
  stage5_scored_top100.parquet
  stage5_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage5 import print_stage5_summary, run
from tracks.shared.paths import ROOT_DIR, RUNTIME_STAGE4_DIR, RUNTIME_STAGE5_DIR

STAGE4_INPUT = RUNTIME_STAGE4_DIR / "stage4_reranked.parquet"
OUTPUT_DIR = RUNTIME_STAGE5_DIR
CONFIG_PATH = ROOT_DIR / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 composite scorer.")
    parser.add_argument(
        "--in",
        dest="stage4_path",
        type=Path,
        default=STAGE4_INPUT,
        help="Stage 4 reranked parquet",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for Stage 5 artifacts",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=CONFIG_PATH,
        help="config.yaml path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = perf_counter()
    result = run(
        stage4_path=args.stage4_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
    )
    print_stage5_summary(result)
    total = perf_counter() - start_time
    print(f"Stage 5 scoring completed in {total:.2f} seconds")


if __name__ == "__main__":
    main()
