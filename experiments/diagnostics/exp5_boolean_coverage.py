"""Experiment 5 — boolean and categorical signal coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from stats_utils import to_float_array


def _append_bool(rows: list[dict], df: pl.DataFrame, col: str, n: int) -> None:
    triggered = int(df[col].cast(pl.Boolean, strict=False).sum())
    pct = triggered / n * 100 if n else 0.0
    flags: list[str] = []
    if pct < 5:
        flags.append("RARELY_TRIGGERED")
    if pct > 95:
        flags.append("ALWAYS_TRIGGERED")
    rows.append(
        {
            "signal_name": col,
            "condition": "== True",
            "count_triggered": triggered,
            "pct_triggered": round(pct, 4),
            "flags": ";".join(flags),
        }
    )


def _append_categorical(
    rows: list[dict],
    df: pl.DataFrame,
    col: str,
    value: str,
    n: int,
) -> None:
    triggered = int((df[col].cast(pl.Utf8) == value).sum())
    pct = triggered / n * 100 if n else 0.0
    flags: list[str] = []
    if pct > 80:
        flags.append("CATEGORY_DOMINANT")
    rows.append(
        {
            "signal_name": col,
            "condition": f"== {value}",
            "count_triggered": triggered,
            "pct_triggered": round(pct, 4),
            "flags": ";".join(flags),
        }
    )


def _append_threshold(
    rows: list[dict],
    df: pl.DataFrame,
    col: str,
    condition: str,
    mask: pl.Series,
    n: int,
) -> None:
    triggered = int(mask.sum())
    pct = triggered / n * 100 if n else 0.0
    rows.append(
        {
            "signal_name": col,
            "condition": condition,
            "count_triggered": triggered,
            "pct_triggered": round(pct, 4),
            "flags": "",
        }
    )


def run_exp5(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    n = df.height
    rows: list[dict] = []

    for col in [
        "in_sweet_spot",
        "stale_coding",
        "has_any_production_role",
        "title_ambiguous",
        "has_github",
    ]:
        _append_bool(rows, df, col, n)

    for value in ["in_band", "near_band"]:
        _append_categorical(rows, df, "exp_band", value, n)

    for value in ["product_heavy", "mixed", "consulting_heavy", "unknown"]:
        _append_categorical(rows, df, "career_type", value, n)

    for value in ["preferred", "acceptable", "outside_india", "unknown"]:
        _append_categorical(rows, df, "location_tier", value, n)

    _append_threshold(
        rows,
        df,
        "title_chasing_penalty",
        "> 0",
        df["title_chasing_penalty"].cast(pl.Float64) > 0,
        n,
    )
    _append_threshold(
        rows,
        df,
        "closed_source_penalty",
        "> 0",
        df["closed_source_penalty"].cast(pl.Float64) > 0,
        n,
    )
    _append_threshold(
        rows,
        df,
        "consulting_resid_penalty",
        "> 0",
        df["consulting_resid_penalty"].cast(pl.Float64) > 0,
        n,
    )
    _append_threshold(
        rows,
        df,
        "ambiguity_penalty",
        "> 0",
        df["ambiguity_penalty"].cast(pl.Float64) > 0,
        n,
    )
    _append_threshold(
        rows,
        df,
        "q3_residual_penalty",
        "> 0.05",
        df["q3_residual_penalty"].cast(pl.Float64) > 0.05,
        n,
    )
    _append_threshold(
        rows,
        df,
        "optional_bonus",
        "> 0",
        df["optional_bonus"].cast(pl.Float64) > 0,
        n,
    )
    _append_threshold(
        rows,
        df,
        "optional_bonus",
        ">= 0.04",
        df["optional_bonus"].cast(pl.Float64) >= 0.04,
        n,
    )
    _append_threshold(
        rows,
        df,
        "optional_bonus",
        ">= 0.08",
        df["optional_bonus"].cast(pl.Float64) >= 0.08,
        n,
    )

    penalty_vals = to_float_array(df["total_penalty"])
    if penalty_vals.size:
        p50, p75, p95 = np.percentile(penalty_vals, [50, 75, 95])
        rows.append(
            {
                "signal_name": "total_penalty",
                "condition": "distribution",
                "count_triggered": int(penalty_vals.size),
                "pct_triggered": round(float(np.mean(penalty_vals)), 4),
                "flags": f"std={float(np.std(penalty_vals)):.4f};p50={p50:.4f};p75={p75:.4f};p95={p95:.4f};max={float(np.max(penalty_vals)):.4f}",
            }
        )

    am = df["availability_multiplier"].cast(pl.Float64)
    at_floor = int((am == 0.5).sum())
    above_09 = int((am > 0.9).sum())
    between_05_07 = int(((am > 0.5) & (am <= 0.7)).sum())
    between_07_09 = int(((am > 0.7) & (am <= 0.9)).sum())

    for condition, count in [
        ("== 0.5 (floor)", at_floor),
        ("> 0.9", above_09),
        ("between 0.5 and 0.7", between_05_07),
        ("between 0.7 and 0.9", between_07_09),
    ]:
        pct = count / n * 100 if n else 0.0
        rows.append(
            {
                "signal_name": "availability_multiplier",
                "condition": condition,
                "count_triggered": count,
                "pct_triggered": round(pct, 4),
                "flags": "",
            }
        )

    consulting_pen = int((df["consulting_resid_penalty"].cast(pl.Float64) > 0).sum())
    consulting_heavy = int((df["career_type"].cast(pl.Utf8) == "consulting_heavy").sum())
    if consulting_pen > 0.5 * n and consulting_heavy < 0.1 * n:
        rows.append(
            {
                "signal_name": "consulting_resid_penalty vs career_type",
                "condition": "PENALTY_MISMATCH",
                "count_triggered": consulting_pen,
                "pct_triggered": round(consulting_heavy / n * 100, 4) if n else 0.0,
                "flags": "PENALTY_MISMATCH",
            }
        )

    result = pl.DataFrame(rows)
    result.write_csv(output_dir / "exp5_boolean_coverage.csv")
    return result
