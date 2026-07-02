#!/usr/bin/env python3
"""
Apply Stage 6 reasoning to a ranking-only CSV.

Reads a ranking CSV (candidate_id, rank, score) from rank.py and writes the
spec-compliant submission file (registered participant ID + reasoning column):

    SignalHunters.csv  →  SignalHunters_reasoning.csv

Usage (from repo root):

    python apply_reasoning.py
    python apply_reasoning.py --input ./SignalHunters.csv
    python apply_reasoning.py --input ./SignalHunters.csv --out ./SignalHunters_reasoning.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage6 import print_stage6_summary, run_from_ranking_csv
from tracks.shared.paths import (
    RANKING_CSV_PATH,
    REASONING_CSV_PATH,
    ROOT_DIR,
    RUNTIME_STAGE6_DIR,
)

CONFIG_PATH = ROOT_DIR / "config.yaml"


def _validate_submission(csv_path: Path) -> None:
    validator = ROOT_DIR / "tools" / "validate_submission.py"
    result = subprocess.run(
        [sys.executable, str(validator), str(csv_path)],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Stage 6 reasoning to a ranking-only CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RANKING_CSV_PATH,
        help=f"Ranking CSV from rank.py (default: {RANKING_CSV_PATH.name})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REASONING_CSV_PATH,
        help=f"Submission CSV with reasoning (default: {REASONING_CSV_PATH.name})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="config.yaml path",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=RUNTIME_STAGE6_DIR,
        help="Directory for Stage 6 audit artifacts",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip tools/validate_submission.py check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.out.resolve()

    if not input_path.exists():
        raise SystemExit(f"Ranking CSV not found: {input_path}")

    started = perf_counter()
    result = run_from_ranking_csv(
        ranking_csv_path=input_path,
        output_csv_path=output_path,
        config_path=args.config,
        output_dir=args.artifacts_dir,
    )
    print_stage6_summary(result)
    print(f"\nReasoning submission CSV: {output_path}")

    if not args.skip_validate:
        _validate_submission(output_path)

    print(f"Stage 6 reasoning completed in {perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
