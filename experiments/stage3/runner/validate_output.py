#!/usr/bin/env python3
"""Validate Stage 3 runner outputs against the local contract."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3.shared.config_runner import DEFAULT_CONFIG, load_runner_config
from experiments.stage3.shared.io_precompute import load_manifest

RUNNER_DIR = Path(__file__).resolve().parent

REQUIRED_SCORE_COLS = frozenset(
    {
        "q1_score",
        "q1_rank",
        "q2_score",
        "q2_rank",
        "skill_score",
        "skill_rank",
        "q3_neg_sim",
        "rrf_score",
        "fused_score",
        "stage3_rank",
    }
)
FORBIDDEN_COLS = frozenset({"bm25_score", "bm25_rank"})


def validate(config_path: Path | None = None) -> None:
    config = load_runner_config(config_path)
    manifest = load_manifest(config.precomputed_manifest)
    output_dir = config.output_dir
    parquet_path = output_dir / "stage3_retrieved.parquet"
    stage2_path = manifest.cohort_stage2

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing {parquet_path}. Run: python experiments/stage3/runner/run.py"
        )

    df = pl.read_parquet(parquet_path)
    stage2_ids = set(
        pl.read_parquet(stage2_path)["candidate_id"].cast(pl.Utf8).to_list()
    )

    errors: list[str] = []

    row_count = df.height
    if row_count < config.min_k or row_count > config.max_k:
        errors.append(
            f"Row count {row_count} outside [{config.min_k}, {config.max_k}]"
        )

    missing_cols = REQUIRED_SCORE_COLS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {sorted(missing_cols)}")

    forbidden = FORBIDDEN_COLS & set(df.columns)
    if forbidden:
        errors.append(f"Forbidden BM25 columns present: {sorted(forbidden)}")

    if df["candidate_id"].n_unique() != df.height:
        errors.append("Duplicate candidate_id in output")

    output_ids = set(df["candidate_id"].cast(pl.Utf8).to_list())
    extra = output_ids - stage2_ids
    if extra:
        errors.append(f"{len(extra)} output IDs not in stage2 cohort")

    ranks = df.sort("stage3_rank")["stage3_rank"].to_list()
    if ranks != list(range(1, row_count + 1)):
        errors.append("stage3_rank is not exactly 1..N")

    sorted_df = df.sort("stage3_rank")
    fused = sorted_df["fused_score"].to_list()
    for i in range(len(fused) - 1):
        if fused[i] < fused[i + 1]:
            errors.append(
                f"fused_score not monotonic with stage3_rank at ranks {i + 1} and {i + 2}"
            )
            break

    for path_name in (
        "stage3_score_distribution.csv",
        "stage3_summary.json",
        "stage3_retrieved.json",
    ):
        if not (output_dir / path_name).exists():
            errors.append(f"Missing output artifact: {path_name}")

    if errors:
        print("VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("VALIDATION PASSED")
    print(f"  Rows: {row_count} (bounds [{config.min_k}, {config.max_k}])")
    print(f"  Columns: skill_score/skill_rank present, no bm25_*")
    print(f"  All output IDs subset of stage2 cohort ({len(stage2_ids)} ids)")


def main() -> None:
    validate()


if __name__ == "__main__":
    main()
