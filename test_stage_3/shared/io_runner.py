"""Runner I/O — load stage2, write stage3 outputs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl

from test_stage_3.shared.config_runner import RunnerConfig

REQUIRED_STAGE2_COLUMNS = frozenset(
    {
        "candidate_id",
        "cluster_id",
        "dist_to_centroid",
        "exp_band",
        "in_sweet_spot",
        "title_family",
        "skill_kw_density",
        "title_ambiguous",
        "stale_profile",
        "low_responder",
        "not_open",
    }
)


def load_stage2_gated(path: Path, config: RunnerConfig) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 output: {path}")

    df = pl.read_parquet(path)
    missing = REQUIRED_STAGE2_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"stage2_gated.parquet missing required columns: {sorted(missing)}")

    count = df.height
    print(f"Stage 2 survivors loaded: {count:,}")
    if count < config.expected_survivor_min or count > config.expected_survivor_max:
        warnings.warn(
            f"Stage 2 survivor count {count} outside expected range "
            f"[{config.expected_survivor_min}, {config.expected_survivor_max}]",
            stacklevel=2,
        )
    return df


def load_skill_features(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate features: {path}")

    df = pl.read_parquet(path, columns=["candidate_id", "skill_weighted_score"])
    if "skill_weighted_score" not in df.columns:
        raise ValueError(
            "skill_weighted_score column not found in candidate_features.parquet — "
            "ensure precompute skill scoring (Blocks A-D) has been run."
        )
    print(f"Loaded skill features: {df.height:,} candidates")
    return df


def _write_retrieved_json(path: Path, df: pl.DataFrame) -> None:
    records = df.to_dicts() if df.height > 0 else []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def write_stage3_outputs(
    output_dir: Path,
    retrieved_df: pl.DataFrame,
    distribution_df: pl.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieved_df.write_parquet(output_dir / "stage3_retrieved.parquet")
    _write_retrieved_json(output_dir / "stage3_retrieved.json", retrieved_df)
    distribution_df.write_csv(output_dir / "stage3_score_distribution.csv")

    with open(output_dir / "stage3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {output_dir / 'stage3_retrieved.parquet'}")
    print(f"Wrote {output_dir / 'stage3_retrieved.json'}")
    print(f"Wrote {output_dir / 'stage3_score_distribution.csv'}")
    print(f"Wrote {output_dir / 'stage3_summary.json'}")
