"""Experiment 4 — availability rank flip analysis (Stage 5 v2 tier3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def _rank_by_column(df: pl.DataFrame, col: str, descending: bool = True) -> pl.Series:
    return df[col].rank(method="average", descending=descending)


def run_exp4(df: pl.DataFrame, output_dir: Path) -> dict:
    pre_avail = df["score_after_t2"].cast(pl.Float64).to_numpy()
    post_avail = df["score_after_t3"].cast(pl.Float64).to_numpy()
    avail_unit = df["avail_unit"].cast(pl.Int64).to_numpy()
    candidate_ids = df["candidate_id"].cast(pl.Utf8).to_list()

    n = len(pre_avail)
    total_pairs = n * (n - 1) // 2

    wrong_flip_count = 0
    small_gap = 0
    medium_gap = 0
    large_gap = 0
    detail_rows: list[dict] = []

    final_rank = _rank_by_column(df, "final_score")

    def _record_flip(i: int, j: int) -> None:
        nonlocal wrong_flip_count, small_gap, medium_gap, large_gap
        wrong_flip_count += 1
        gap = pre_avail[i] - pre_avail[j]
        if gap < 0.05:
            small_gap += 1
        elif gap <= 0.15:
            medium_gap += 1
        else:
            large_gap += 1

        if gap > 0.10:
            detail_rows.append(
                {
                    "candidate_i": candidate_ids[i],
                    "candidate_j": candidate_ids[j],
                    "pre_avail_i": float(pre_avail[i]),
                    "pre_avail_j": float(pre_avail[j]),
                    "pre_avail_gap": float(gap),
                    "avail_unit_i": int(avail_unit[i]),
                    "avail_unit_j": int(avail_unit[j]),
                    "avail_unit_gap": int(avail_unit[i] - avail_unit[j]),
                    "final_rank_i": int(final_rank[i]),
                    "final_rank_j": int(final_rank[j]),
                }
            )

    for i in range(n):
        for j in range(i + 1, n):
            if pre_avail[i] > pre_avail[j] and post_avail[i] < post_avail[j]:
                _record_flip(i, j)
            elif pre_avail[j] > pre_avail[i] and post_avail[j] < post_avail[i]:
                _record_flip(j, i)

    candidates_at_tier_c = int(np.sum(avail_unit == -1))

    top100_final = set(
        df.sort("final_score", descending=True).head(100)["candidate_id"].cast(pl.Utf8).to_list()
    )
    top100_pre_avail = set(
        df.sort("score_after_t2", descending=True).head(100)["candidate_id"].cast(pl.Utf8).to_list()
    )

    displaced = len(top100_pre_avail - top100_final)
    rescued = len(top100_final - top100_pre_avail)

    pct = (wrong_flip_count / total_pairs * 100) if total_pairs else 0.0

    summary_rows = [
        {"metric": "total_pairs_compared", "value": total_pairs},
        {"metric": "wrong_flip_count", "value": wrong_flip_count},
        {"metric": "wrong_flip_percentage", "value": round(pct, 4)},
        {"metric": "small_gap_flips", "value": small_gap},
        {"metric": "medium_gap_flips", "value": medium_gap},
        {"metric": "large_gap_flips", "value": large_gap},
        {"metric": "candidates_at_avail_tier_c", "value": candidates_at_tier_c},
        {"metric": "top100_candidates_displaced", "value": displaced},
        {"metric": "top100_candidates_rescued", "value": rescued},
    ]

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(output_dir / "exp4_availability_flips.csv")

    if detail_rows:
        detail_df = pl.DataFrame(detail_rows).sort("pre_avail_gap", descending=True)
    else:
        detail_df = pl.DataFrame(
            schema={
                "candidate_i": pl.Utf8,
                "candidate_j": pl.Utf8,
                "pre_avail_i": pl.Float64,
                "pre_avail_j": pl.Float64,
                "pre_avail_gap": pl.Float64,
                "avail_unit_i": pl.Int64,
                "avail_unit_j": pl.Int64,
                "avail_unit_gap": pl.Int64,
                "final_rank_i": pl.Int64,
                "final_rank_j": pl.Int64,
            }
        )
    detail_df.write_csv(output_dir / "exp4_large_gap_flips_detail.csv")

    return {
        "summary": summary_df,
        "detail": detail_df,
        "summary_dict": {r["metric"]: r["value"] for r in summary_rows},
    }
