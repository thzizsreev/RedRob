#!/usr/bin/env python3
"""Stage 5 diagnostic experiments — read-only analysis of stage5_scored.parquet."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIAG_ROOT = Path(__file__).resolve().parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_DIAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIAG_ROOT))

import polars as pl

from tracks.shared.paths import CANDIDATES_JSONL_PATH, RUNTIME_STAGE5_DIR

from avail_factors import AvailConfig
from columns import enrich_scored_df, validate_columns
from exp1_signal_variance import run_exp1
from exp2_correlation import run_exp2
from exp3_layer_rank_stability import run_exp3
from exp4_availability_flips import run_exp4
from exp5_boolean_coverage import run_exp5
from exp6_avail_subfactors import run_exp6
from json_output import build_diagnostics_json, write_json_outputs
from report import print_stdout_summary, write_report

# --- Input / output paths (edit here) ---
SCORED_PARQUET = RUNTIME_STAGE5_DIR / "stage5_scored.parquet"
CANDIDATES_JSONL = CANDIDATES_JSONL_PATH
OUTPUT_DIR = _DIAG_ROOT / "output"

# Exp6 must match Stage 5 scoring date (config.yaml stage5.current_date)
CURRENT_DATE = date(2026, 6, 22)

AVAIL_CONFIG = AvailConfig(
    good_response_rate=0.5,
    response_floor=0.6,
    slow_response_hours=24.0,
    response_decay_window_hours=168.0,
    speed_floor=0.7,
    fresh_days=30,
    recency_decay_window=180,
    recency_floor=0.6,
    not_open_factor=0.85,
    interview_floor=0.7,
    offer_floor=0.8,
    market_inactive_factor=0.95,
    avail_min=0.5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 diagnostic experiments")
    parser.add_argument(
        "--scored",
        type=Path,
        default=SCORED_PARQUET,
        help=f"Path to stage5_scored.parquet (default: {SCORED_PARQUET})",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=CANDIDATES_JSONL,
        help=f"Path to candidates JSONL/JSON (default: {CANDIDATES_JSONL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored_path: Path = args.scored
    jsonl_path: Path = args.jsonl
    output_dir: Path = args.out

    if not scored_path.exists():
        print(f"Missing scored parquet: {scored_path}", file=sys.stderr)
        print("Run Stage 5 first: python tracks/instructor/stage5/run.py", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {scored_path}")
    df = pl.read_parquet(scored_path)
    print(f"Loaded {df.height:,} candidates")

    validate_columns(df)
    df = enrich_scored_df(df)

    print("Running Experiment 1 — signal variance")
    exp1 = run_exp1(df, output_dir)

    print("Running Experiment 2 — correlation")
    exp2 = run_exp2(df, output_dir)

    print("Running Experiment 3 — layer rank stability")
    exp3 = run_exp3(df, output_dir)

    print("Running Experiment 4 — availability rank flips")
    exp4 = run_exp4(df, output_dir)

    print("Running Experiment 5 — boolean coverage")
    exp5 = run_exp5(df, output_dir)

    print("Running Experiment 6 — availability sub-factors")
    exp6 = run_exp6(df, jsonl_path, output_dir, CURRENT_DATE, AVAIL_CONFIG)

    print("Writing combined report")
    report_path, section7 = write_report(
        output_dir, exp1, exp2, exp3, exp4, exp5, exp6, df
    )
    print(f"Wrote {report_path}")

    print("Writing JSON outputs")
    json_payload = build_diagnostics_json(
        scored_path=scored_path,
        jsonl_path=jsonl_path,
        candidate_count=df.height,
        current_date=CURRENT_DATE,
        exp1=exp1,
        exp2=exp2,
        exp3=exp3,
        exp4=exp4,
        exp5=exp5,
        exp6=exp6,
        section7=section7,
    )
    json_path = write_json_outputs(output_dir, json_payload)
    print(f"Wrote {json_path}")
    print(f"Wrote per-experiment JSON files (exp1..exp6) in {output_dir}")

    print_stdout_summary(exp1, exp2, exp3, section7)
    print(f"\nAll outputs written to {output_dir}")


if __name__ == "__main__":
    main()
