#!/usr/bin/env python3
"""
Stage 6 — 3-sentence reasoning builder (CPU ONNX paraphraser).

Run after Stage 5:
    python tracks/instructor/stage0/run_paraphraser_export.py
    python tracks/instructor/stage0/run_reasoning_raw_precompute.py
    python tracks/instructor/stage5/run.py
    python tracks/instructor/stage6/run.py

Outputs (under artifacts/runtime/stage6/):
  {team_id}.csv
  stage6_reasoning.parquet
  stage6_summary.json

Also copies the final CSV to ./SignalHunters_reasoning.csv in the repo root.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage6 import print_stage6_summary, run
from tracks.shared.paths import ROOT_DIR, RUNTIME_STAGE6_DIR, REASONING_CSV_PATH

CONFIG_PATH = ROOT_DIR / "config.yaml"
OUTPUT_DIR = RUNTIME_STAGE6_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 reasoning builder.")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=CONFIG_PATH,
        help="config.yaml path",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for Stage 6 artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = perf_counter()
    result = run(config_path=args.config_path, output_dir=args.output_dir)
    print_stage6_summary(result)
    shutil.copy2(result.csv_path, REASONING_CSV_PATH)
    print(f"Submission CSV (repo root): {REASONING_CSV_PATH.resolve()}")
    print(f"Stage 6 completed in {perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()
