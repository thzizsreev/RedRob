"""Parquet column contract for Stage 5 diagnostics."""

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
        "ce_norm",
        "fused_norm",
        "q1_norm",
        "q2_norm",
        "q3_norm",
        "core",
        "core_floored",
        "shaped",
        "penalized",
        "bonused",
        "availability_adj",
        "final_score",
        "keyword_ratio",
        "assessment_cov",
        "combined_coverage",
        "must_have_floor_multiplier",
        "product_company_fraction",
        "in_sweet_spot",
        "exp_band",
        "stale_coding",
        "has_any_production_role",
        "shape_mult",
        "title_chasing_penalty",
        "q3_residual_penalty",
        "closed_source_penalty",
        "ambiguity_penalty",
        "consulting_resid_penalty",
        "total_penalty",
        "short_hop_count",
        "title_ambiguous",
        "career_type",
        "external_validation_score",
        "has_github",
        "optional_bonus",
        "availability_multiplier",
        "location_adj",
        "workmode_adj",
        "notice_adj",
        "logistics_adjustment",
        "location_tier",
        "notice_period_days",
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
    "ce_norm",
    "fused_norm",
    "q1_norm",
    "q2_norm",
    "q3_norm",
    "core",
    "core_floored",
    "shaped",
    "penalized",
    "bonused",
    "availability_adj",
    "final_score",
]

GROUP_C_SIGNALS = [
    "product_company_fraction",
    "external_validation_score",
    "must_have_floor_multiplier",
    "shape_mult",
    "total_penalty",
    "optional_bonus",
    "availability_multiplier",
]

NORMALIZED_SIGNALS = frozenset(
    {"ce_norm", "fused_norm", "q1_norm", "q2_norm", "q3_norm"}
)

RETRIEVAL_CORRELATION_SIGNALS = [
    "ce_norm",
    "fused_norm",
    "q1_norm",
    "q2_norm",
    "q3_norm",
    "skill_score",
]

ALL_CORRELATION_SIGNALS = GROUP_A_SIGNALS + GROUP_B_SIGNALS + GROUP_C_SIGNALS

LAYER_TRANSITIONS = [
    ("L0→L1", "ce_norm", "core"),
    ("L1→L2", "core", "core_floored"),
    ("L2→L3", "core_floored", "shaped"),
    ("L3→L4", "shaped", "penalized"),
    ("L4→L5", "penalized", "bonused"),
    ("L5→L6", "bonused", "availability_adj"),
    ("L6→L7", "availability_adj", "final_score"),
]


def validate_columns(df: pl.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        print("Missing required columns in stage5_scored.parquet:", file=sys.stderr)
        for col in missing:
            print(f"  - {col}", file=sys.stderr)
        sys.exit(1)
