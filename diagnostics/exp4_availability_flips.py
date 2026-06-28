"""Experiment 4 — availability rank flip analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def _rank_by_column(df: pl.DataFrame, col: str, descending: bool = True) -> pl.Series:
    return df[col].rank(method="average", descending=descending)


def run_exp4(df: pl.DataFrame, output_dir: Path) -> dict:
    bonused = df["bonused"].cast(pl.Float64).to_numpy()
    avail_adj = df["availability_adj"].cast(pl.Float64).to_numpy()
    avail_mult = df["availability_multiplier"].cast(pl.Float64).to_numpy()
    candidate_ids = df["candidate_id"].cast(pl.Utf8).to_list()

    n = len(bonused)
    total_pairs = n * (n - 1) // 2

    wrong_flip_count = 0
    small_gap = 0
    medium_gap = 0
    large_gap = 0
    detail_rows: list[dict] = []

    final_rank = _rank_by_column(df, "final_score")

    for i in range(n):
        for j in range(i + 1, n):
            if bonused[i] > bonused[j] and avail_adj[i] < avail_adj[j]:
                wrong_flip_count += 1
                gap = bonused[i] - bonused[j]
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
                            "bonused_i": float(bonused[i]),
                            "bonused_j": float(bonused[j]),
                            "bonused_gap": float(gap),
                            "avail_mult_i": float(avail_mult[i]),
                            "avail_mult_j": float(avail_mult[j]),
                            "avail_gap": float(avail_mult[i] - avail_mult[j]),
                            "final_rank_i": int(final_rank[i]),
                            "final_rank_j": int(final_rank[j]),
                        }
                    )
            elif bonused[j] > bonused[i] and avail_adj[j] < avail_adj[i]:
                wrong_flip_count += 1
                gap = bonused[j] - bonused[i]
                if gap < 0.05:
                    small_gap += 1
                elif gap <= 0.15:
                    medium_gap += 1
                else:
                    large_gap += 1

                if gap > 0.10:
                    detail_rows.append(
                        {
                            "candidate_i": candidate_ids[j],
                            "candidate_j": candidate_ids[i],
                            "bonused_i": float(bonused[j]),
                            "bonused_j": float(bonused[i]),
                            "bonused_gap": float(gap),
                            "avail_mult_i": float(avail_mult[j]),
                            "avail_mult_j": float(avail_mult[i]),
                            "avail_gap": float(avail_mult[j] - avail_mult[i]),
                            "final_rank_i": int(final_rank[j]),
                            "final_rank_j": int(final_rank[i]),
                        }
                    )

    candidates_at_floor = int(np.sum(avail_mult == 0.5))

    top100_final = set(
        df.sort("final_score", descending=True).head(100)["candidate_id"].cast(pl.Utf8).to_list()
    )
    top100_bonused = set(
        df.sort("bonused", descending=True).head(100)["candidate_id"].cast(pl.Utf8).to_list()
    )

    displaced = len(top100_bonused - top100_final)
    rescued = len(top100_final - top100_bonused)

    pct = (wrong_flip_count / total_pairs * 100) if total_pairs else 0.0

    summary_rows = [
        {"metric": "total_pairs_compared", "value": total_pairs},
        {"metric": "wrong_flip_count", "value": wrong_flip_count},
        {"metric": "wrong_flip_percentage", "value": round(pct, 4)},
        {"metric": "small_gap_flips", "value": small_gap},
        {"metric": "medium_gap_flips", "value": medium_gap},
        {"metric": "large_gap_flips", "value": large_gap},
        {"metric": "candidates_at_avail_floor", "value": candidates_at_floor},
        {"metric": "top100_candidates_displaced", "value": displaced},
        {"metric": "top100_candidates_rescued", "value": rescued},
    ]

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(output_dir / "exp4_availability_flips.csv")

    if detail_rows:
        detail_df = pl.DataFrame(detail_rows).sort("bonused_gap", descending=True)
    else:
        detail_df = pl.DataFrame(
            schema={
                "candidate_i": pl.Utf8,
                "candidate_j": pl.Utf8,
                "bonused_i": pl.Float64,
                "bonused_j": pl.Float64,
                "bonused_gap": pl.Float64,
                "avail_mult_i": pl.Float64,
                "avail_mult_j": pl.Float64,
                "avail_gap": pl.Float64,
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
