#!/usr/bin/env python3
"""
Stages 1–5 ranking pipeline — run the full instructor ranking procedure.

Requires Stage 0 precompute artifacts on disk (vectors, FAISS, BM25, clusters).
Does not run Stage 0 or cross-encoder export.

Individual stages can still be run standalone:
    python tracks/instructor/stage1/run_filter.py
    python tracks/instructor/stage2/run.py
    python tracks/instructor/stage3/run.py
    python tracks/instructor/stage4/run.py
    python tracks/instructor/stage5/run.py

Full pipeline:
    python tracks/instructor/run_pipeline.py

Importable API (e.g. resume ranking integration):
    from tracks.instructor.pipeline import RankingPipelineConfig, run_ranking_pipeline
    result = run_ranking_pipeline()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.core.config import STAGE1_RANDOM_SEED
from tracks.instructor.pipeline import RankingPipelineConfig, run_ranking_pipeline
from tracks.shared.paths import (
    CANDIDATES_JSONL_PATH,
    ROOT_DIR,
    RUNTIME_STAGE0_DIR,
    RUNTIME_STAGE1_DIR,
    RUNTIME_STAGE2_DIR,
    RUNTIME_STAGE3_DIR,
    RUNTIME_STAGE4_DIR,
    RUNTIME_STAGE5_DIR,
)


def _runtime_dirs(runtime_dir: Path) -> dict[str, Path]:
    return {
        "stage0_path": runtime_dir / "stage0",
        "stage1_path": runtime_dir / "stage1",
        "stage2_output_dir": runtime_dir / "stage2",
        "stage3_output_dir": runtime_dir / "stage3",
        "stage4_output_dir": runtime_dir / "stage4",
        "stage5_output_dir": runtime_dir / "stage5",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run instructor ranking pipeline (stages 1–5)."
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=ROOT_DIR / "config.yaml",
        help="config.yaml path",
    )
    parser.add_argument(
        "--candidates",
        dest="candidates_path",
        type=Path,
        default=CANDIDATES_JSONL_PATH,
        help="Candidates JSONL path",
    )
    parser.add_argument(
        "--runtime-dir",
        dest="runtime_dir",
        type=Path,
        default=None,
        help="Shortcut: set all artifacts/runtime/stageN paths from one parent dir",
    )
    parser.add_argument(
        "--stage0",
        dest="stage0_path",
        type=Path,
        default=None,
        help="Stage 0 artifacts directory",
    )
    parser.add_argument(
        "--stage1",
        dest="stage1_path",
        type=Path,
        default=None,
        help="Stage 1 artifacts directory",
    )
    parser.add_argument(
        "--out-stage2",
        dest="stage2_output_dir",
        type=Path,
        default=None,
        help="Stage 2 output directory",
    )
    parser.add_argument(
        "--out-stage3",
        dest="stage3_output_dir",
        type=Path,
        default=None,
        help="Stage 3 output directory",
    )
    parser.add_argument(
        "--out-stage4",
        dest="stage4_output_dir",
        type=Path,
        default=None,
        help="Stage 4 output directory",
    )
    parser.add_argument(
        "--out-stage5",
        dest="stage5_output_dir",
        type=Path,
        default=None,
        help="Stage 5 output directory",
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        type=int,
        default=STAGE1_RANDOM_SEED,
        help="Stage 1 cluster filter random seed",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-stage summaries (timing table still prints)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RankingPipelineConfig:
    if args.runtime_dir is not None:
        dirs = _runtime_dirs(args.runtime_dir)
    else:
        dirs = {
            "stage0_path": RUNTIME_STAGE0_DIR,
            "stage1_path": RUNTIME_STAGE1_DIR,
            "stage2_output_dir": RUNTIME_STAGE2_DIR,
            "stage3_output_dir": RUNTIME_STAGE3_DIR,
            "stage4_output_dir": RUNTIME_STAGE4_DIR,
            "stage5_output_dir": RUNTIME_STAGE5_DIR,
        }

    overrides = {
        "stage0_path": args.stage0_path,
        "stage1_path": args.stage1_path,
        "stage2_output_dir": args.stage2_output_dir,
        "stage3_output_dir": args.stage3_output_dir,
        "stage4_output_dir": args.stage4_output_dir,
        "stage5_output_dir": args.stage5_output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            dirs[key] = value

    return RankingPipelineConfig(
        **dirs,
        candidates_path=args.candidates_path,
        config_path=args.config_path,
        random_seed=args.random_seed,
        print_summaries=not args.quiet,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    result = run_ranking_pipeline(config)
    print(f"\nFinal submission: {result.final_csv_path}")


if __name__ == "__main__":
    main()
