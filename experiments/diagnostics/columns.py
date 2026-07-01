"""Parquet column contract for Stage 5 v2 diagnostics."""

from __future__ import annotations

import sys

import polars as pl

REQUIRED_COLUMNS = frozenset(
    {
        "candidate_id",
        "stage3_rank",
        "stage4_rank",
        "cross_encoder_score",
        "fused_score",
        "q1_score",
        "q2_score",
        "q3_neg_sim",
        "skill_score",
        "rank_ce",
        "rank_q1",
        "rank_q2",
        "borda_sum",
        "borda_primary",
        "t1_std",
        "sweet_bonus",
        "tier2_raw",
        "tier2_scaled",
        "t2_std",
        "avail_tier",
        "avail_unit",
        "tier3_scaled",
        "t3_std",
        "location_unit",
        "workmode_unit",
        "notice_unit",
        "tier4_raw",
        "tier4_scaled",
        "t4_std",
        "final_score",
        "product_company_fraction",
        "in_sweet_spot",
        "exp_band",
        "stale_coding",
        "has_any_production_role",
        "title_chasing_penalty",
        "closed_source_penalty",
        "ambiguity_penalty",
        "optional_bonus",
        "short_hop_count",
        "title_ambiguous",
        "career_type",
        "external_validation_score",
        "has_github",
        "location_tier",
        "notice_period_days",
        "days_since_active",
    }
)

GROUP_A_SIGNALS = [
    "cross_encoder_score",
    "fused_score",
    "q1_score",
    "q2_score",
    "q3_neg_sim",
    "skill_score",
]

GROUP_B_SIGNALS = [
    "borda_sum",
    "borda_primary",
    "tier2_scaled",
    "tier3_scaled",
    "tier4_scaled",
    "final_score",
]

GROUP_C_SIGNALS = [
    "product_company_fraction",
    "external_validation_score",
    "sweet_bonus",
    "optional_bonus",
    "title_chasing_penalty",
    "ambiguity_penalty",
    "closed_source_penalty",
    "avail_unit",
    "location_unit",
    "workmode_unit",
    "notice_unit",
    "tier4_raw",
]

NORMALIZED_SIGNALS = frozenset(
    {"borda_primary", "tier2_scaled", "tier3_scaled", "tier4_scaled"}
)

RETRIEVAL_CORRELATION_SIGNALS = [
    "cross_encoder_score",
    "fused_score",
    "q1_score",
    "q2_score",
    "q3_neg_sim",
    "skill_score",
    "borda_primary",
]

ALL_CORRELATION_SIGNALS = GROUP_A_SIGNALS + GROUP_B_SIGNALS + GROUP_C_SIGNALS

LAYER_TRANSITIONS = [
    ("T0→T1", "borda_sum", "borda_primary"),
    ("T1→T2", "borda_primary", "score_after_t2"),
    ("T2→T3", "score_after_t2", "score_after_t3"),
    ("T3→T4", "score_after_t3", "final_score"),
]


def enrich_scored_df(df: pl.DataFrame) -> pl.DataFrame:
    """Add cumulative tier scores used by layer-stability and availability experiments."""
    return df.with_columns(
        [
            (pl.col("borda_primary") + pl.col("tier2_scaled")).alias("score_after_t2"),
            (
                pl.col("borda_primary")
                + pl.col("tier2_scaled")
                + pl.col("tier3_scaled")
            ).alias("score_after_t3"),
        ]
    )


def validate_columns(df: pl.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        print("Missing required columns in stage5_scored.parquet:", file=sys.stderr)
        for col in missing:
            print(f"  - {col}", file=sys.stderr)
        sys.exit(1)
