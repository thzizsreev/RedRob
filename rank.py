#!/usr/bin/env python3
"""
Hackathon ranking entry point (Stages 1–5, CPU-only).

Produces a validated top-100 ranking CSV (candidate_id, rank, score only) from
precomputed Stage 0 artifacts. Use apply_reasoning.py for the reasoning column.
Precompute is offline (no time limit); this script is the bounded ranking step.

Spec reproduce command (from repo root):

    python rank.py --candidates ./data/candidates.jsonl --out ./SignalHunters.csv

Defaults write ./SignalHunters.csv in the repo root:

    python rank.py

Requires artifacts under artifacts/runtime/stage0 and stage1 (see README).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.pipeline import RankingPipelineConfig, run_ranking_pipeline
from tracks.instructor.stage5.validate import validate_ranking_csv
from tracks.shared.paths import (
    CANDIDATES_JSONL_PATH,
    RANKING_CSV_PATH,
    ROOT_DIR,
    RUNTIME_STAGE0_DIR,
    TEAM_ID,
)

TIME_BUDGET_SECONDS = 300


def _validate_ranking(csv_path: Path) -> None:
    try:
        validate_ranking_csv(csv_path, expected_rows=100)
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("Ranking CSV is valid.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidates (Stages 1–5) and write ranking CSV (no reasoning)."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=CANDIDATES_JSONL_PATH,
        help="Path to candidates.jsonl (default: data/candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RANKING_CSV_PATH,
        help=f"Output ranking CSV (default: ./{TEAM_ID}.csv in repo root)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config.yaml",
        help="Pipeline config.yaml",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=RUNTIME_STAGE0_DIR.parent,
        help="Parent of stage0 … stage5 runtime artifacts",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip ranking CSV validation",
    )
    return parser.parse_args()


def run(
    *,
    candidates_path: Path = CANDIDATES_JSONL_PATH,
    output_path: Path = RANKING_CSV_PATH,
    config_path: Path | None = None,
    runtime_dir: Path | None = None,
    skip_validate: bool = False,
) -> Path:
    """Programmatic entry point matching the hackathon spec."""
    config_path = config_path or ROOT_DIR / "config.yaml"
    runtime = runtime_dir or RUNTIME_STAGE0_DIR.parent
    os.environ["REDROB_CPU_ONLY"] = "1"

    stage0 = runtime / "stage0"
    stage1 = runtime / "stage1"
    if not (stage0 / "candidate_index.faiss").exists():
        raise FileNotFoundError(
            f"Missing FAISS index at {stage0}. "
            "Run Stage 0 precompute first (see guide/stage0-precompute.md)."
        )
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    started = perf_counter()
    config = RankingPipelineConfig(
        stage0_path=stage0,
        stage1_path=stage1,
        stage2_output_dir=runtime / "stage2",
        stage3_output_dir=runtime / "stage3",
        stage4_output_dir=runtime / "stage4",
        stage5_output_dir=runtime / "stage5",
        candidates_path=candidates_path,
        config_path=config_path,
        print_summaries=True,
        include_reasoning=False,
    )

    result = run_ranking_pipeline(config)
    elapsed = perf_counter() - started

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.final_csv_path, output_path)
    print(f"\nRanking CSV: {output_path}")

    if not skip_validate:
        _validate_ranking(output_path)

    print(f"\nRanking time: {elapsed:.1f}s")
    if elapsed > TIME_BUDGET_SECONDS:
        raise RuntimeError(
            f"Exceeded {TIME_BUDGET_SECONDS}s hackathon CPU budget ({elapsed:.1f}s)"
        )

    if output_path.name != RANKING_CSV_PATH.name:
        print(
            f"Note: output filename ({output_path.name}) differs from default "
            f"({RANKING_CSV_PATH.name}).",
            file=sys.stderr,
        )
    return output_path


def main() -> None:
    args = parse_args()
    os.environ["REDROB_CPU_ONLY"] = "1"
    try:
        run(
            candidates_path=args.candidates,
            output_path=args.out,
            config_path=args.config,
            runtime_dir=args.runtime_dir,
            skip_validate=args.skip_validate,
        )
    except RuntimeError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
