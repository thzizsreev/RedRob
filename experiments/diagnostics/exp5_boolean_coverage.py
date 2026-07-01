"""Experiment 5 — boolean and categorical signal coverage."""

from __future__ import annotations

from pathlib import Path

import polars as pl


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

    for value in ["A", "B", "C"]:
        _append_categorical(rows, df, "avail_tier", value, n)

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
        "ambiguity_penalty",
        "> 0",
        df["ambiguity_penalty"].cast(pl.Float64) > 0,
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
        "sweet_bonus",
        "> 0",
        df["sweet_bonus"].cast(pl.Float64) > 0,
        n,
    )
    _append_threshold(
        rows,
        df,
        "tier2_scaled",
        "< 0",
        df["tier2_scaled"].cast(pl.Float64) < 0,
        n,
    )

    avail_unit = df["avail_unit"].cast(pl.Int64)
    tier_c = int((avail_unit == -1).sum())
    tier_b = int((avail_unit == 0).sum())
    tier_a = int((avail_unit == 1).sum())

    for condition, count in [
        ("== -1 (tier C)", tier_c),
        ("== 0 (tier B)", tier_b),
        ("== 1 (tier A)", tier_a),
    ]:
        pct = count / n * 100 if n else 0.0
        rows.append(
            {
                "signal_name": "avail_unit",
                "condition": condition,
                "count_triggered": count,
                "pct_triggered": round(pct, 4),
                "flags": "",
            }
        )

    tier3 = df["tier3_scaled"].cast(pl.Float64)
    tier3_negative = int((tier3 < 0).sum())
    tier3_zero = int((tier3 == 0).sum())
    tier3_positive = int((tier3 > 0).sum())
    for condition, count in [
        ("< 0", tier3_negative),
        ("== 0", tier3_zero),
        ("> 0", tier3_positive),
    ]:
        pct = count / n * 100 if n else 0.0
        rows.append(
            {
                "signal_name": "tier3_scaled",
                "condition": condition,
                "count_triggered": count,
                "pct_triggered": round(pct, 4),
                "flags": "",
            }
        )

    result = pl.DataFrame(rows)
    result.write_csv(output_dir / "exp5_boolean_coverage.csv")
    return result
