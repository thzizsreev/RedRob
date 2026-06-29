"""Experiment 1 — signal variance analysis."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stats_utils import (
    compute_signal_stats,
    flag_variance_stats,
    signal_groups,
    to_float_array,
    write_csv,
)


def run_exp1(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    rows: list[dict] = []
    for signal_name, group in signal_groups():
        values = to_float_array(df[signal_name])
        stats = compute_signal_stats(values)
        flags = flag_variance_stats(signal_name, group, stats)
        rows.append(
            {
                "signal_name": signal_name,
                "group": group,
                **stats,
                "flags": ";".join(flags),
            }
        )

    result = pl.DataFrame(rows)
    out_path = output_dir / "exp1_signal_variance.csv"
    result.write_csv(out_path)

    sorted_df = result.sort("std", descending=True, nulls_last=True)
    sorted_path = output_dir / "exp1_signal_variance_sorted_by_std.csv"
    sorted_df.write_csv(sorted_path)

    return result
