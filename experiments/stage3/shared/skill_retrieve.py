"""L3 skill-score retrieval from precomputed skill_weighted_score."""

from __future__ import annotations

import polars as pl

from experiments.stage3.shared.config_runner import RunnerConfig


def skill_retrieve_l3(
    skill_features: pl.DataFrame,
    stage2_df: pl.DataFrame,
    config: RunnerConfig,
) -> pl.DataFrame:
    survivor_ids = stage2_df.select("candidate_id").unique()

    filtered = (
        skill_features.join(survivor_ids, on="candidate_id", how="inner")
        .sort("skill_weighted_score", descending=True)
        .head(config.per_query_k_skill)
    )

    if filtered.height == 0:
        return pl.DataFrame(
            schema={
                "candidate_id": pl.Utf8,
                "skill_score": pl.Float64,
                "skill_rank": pl.Int64,
            }
        )

    return filtered.with_columns(
        pl.col("skill_weighted_score").alias("skill_score"),
        pl.int_range(1, pl.len() + 1).alias("skill_rank"),
    ).select("candidate_id", "skill_score", "skill_rank")
