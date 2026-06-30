"""Experiment 2 — pairwise correlation matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from columns import ALL_CORRELATION_SIGNALS
from stats_utils import (
    correlation_pair_flags,
    safe_pearson,
    safe_spearman,
    to_float_array,
)


def _correlation_matrix(df: pl.DataFrame, signals: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    arrays = {sig: to_float_array(df[sig]) for sig in signals}
    n = len(signals)
    pearson_rows: list[dict] = []
    spearman_rows: list[dict] = []

    for i, sig_a in enumerate(signals):
        row_p: dict = {"signal": sig_a}
        row_s: dict = {"signal": sig_a}
        for j, sig_b in enumerate(signals):
            min_len = min(len(arrays[sig_a]), len(arrays[sig_b]))
            x = arrays[sig_a][:min_len]
            y = arrays[sig_b][:min_len]
            row_p[sig_b] = safe_pearson(x, y)
            row_s[sig_b] = safe_spearman(x, y)
        pearson_rows.append(row_p)
        spearman_rows.append(row_s)

    return pl.DataFrame(pearson_rows), pl.DataFrame(spearman_rows)


def _high_correlation_pairs(
    df: pl.DataFrame,
    signals: list[str],
) -> pl.DataFrame:
    pairs: list[dict] = []
    for i, sig_a in enumerate(signals):
        x = to_float_array(df[sig_a])
        for j in range(i + 1, len(signals)):
            sig_b = signals[j]
            y = to_float_array(df[sig_b])
            min_len = min(len(x), len(y))
            xa = x[:min_len]
            yb = y[:min_len]
            pearson = safe_pearson(xa, yb)
            spearman = safe_spearman(xa, yb)
            if spearman is None or abs(spearman) <= 0.50:
                continue
            flags = correlation_pair_flags(sig_a, sig_b, pearson, spearman)
            pairs.append(
                {
                    "signal_a": sig_a,
                    "signal_b": sig_b,
                    "pearson": pearson,
                    "spearman": spearman,
                    "flag": ";".join(flags),
                }
            )

    if not pairs:
        return pl.DataFrame(
            schema={
                "signal_a": pl.Utf8,
                "signal_b": pl.Utf8,
                "pearson": pl.Float64,
                "spearman": pl.Float64,
                "flag": pl.Utf8,
            }
        )

    return pl.DataFrame(pairs).sort(pl.col("spearman").abs(), descending=True)


def run_exp2(df: pl.DataFrame, output_dir: Path) -> dict:
    pearson_df, spearman_df = _correlation_matrix(df, ALL_CORRELATION_SIGNALS)
    pearson_df.write_csv(output_dir / "exp2_pearson_matrix.csv")
    spearman_df.write_csv(output_dir / "exp2_spearman_matrix.csv")

    pairs_df = _high_correlation_pairs(df, ALL_CORRELATION_SIGNALS)
    pairs_df.write_csv(output_dir / "exp2_high_correlation_pairs.csv")

    return {
        "pearson_matrix": pearson_df,
        "spearman_matrix": spearman_df,
        "high_correlation_pairs": pairs_df,
    }
