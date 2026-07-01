"""Experiment 6 — availability sub-factor decomposition."""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from avail_factors import (
    FACTOR_FLOORS,
    FACTOR_NAMES,
    AvailConfig,
    availability_multiplier,
    compute_all_factors_with_missing,
)
from jsonl_loader import load_signals_for_ids
from stats_utils import compute_signal_stats, to_float_array


def _flag_subfactor(
    factor_name: str,
    stats: dict,
    count_at_floor: int,
    count_at_ceiling: int,
    count_missing: int,
    n: int,
) -> list[str]:
    flags: list[str] = []
    std = stats.get("std")
    if std is not None and std < 0.03:
        flags.append("NEAR_CONSTANT")
    if n and count_at_floor / n > 0.30:
        flags.append("FLOOR_HEAVY")
    if n and count_at_ceiling / n > 0.70:
        flags.append("CEILING_HEAVY")
    if n and count_missing / n > 0.40:
        flags.append("MISSING_DOMINATED")
    return flags


def _count_at_floor(values: np.ndarray, factor_name: str) -> int:
    floor = FACTOR_FLOORS[factor_name]
    return int(np.sum(np.isclose(values, floor)))


def run_exp6(
    df: pl.DataFrame,
    candidates_path: Path,
    output_dir: Path,
    current_date: date,
    cfg: AvailConfig,
) -> dict:
    candidate_ids = set(df["candidate_id"].cast(pl.Utf8).to_list())
    signals_by_id = load_signals_for_ids(candidates_path, candidate_ids)

    factor_rows: list[dict[str, float]] = []
    missing_counts = {name: 0 for name in FACTOR_NAMES}
    mismatch_count = 0

    if "availability_multiplier" in df.columns:
        parquet_am = {
            str(row["candidate_id"]): float(row["availability_multiplier"])
            for row in df.select("candidate_id", "availability_multiplier").iter_rows(named=True)
        }
    else:
        parquet_am = None

    for cid in df["candidate_id"].cast(pl.Utf8).to_list():
        signals = signals_by_id[cid]
        factors, missing = compute_all_factors_with_missing(signals, current_date, cfg)
        recomputed = availability_multiplier(factors, cfg)
        if parquet_am is not None:
            expected = parquet_am[cid]
            if abs(recomputed - expected) > 0.001:
                mismatch_count += 1
                warnings.warn(
                    f"Availability mismatch for {cid}: recomputed={recomputed:.4f}, "
                    f"parquet={expected:.4f}",
                    stacklevel=2,
                )

        for name in FACTOR_NAMES:
            if missing[name]:
                missing_counts[name] += 1
        factor_rows.append(factors)

    if mismatch_count:
        print(
            f"WARNING: {mismatch_count} availability_multiplier mismatches (>0.001 tolerance)",
            file=sys.stderr,
        )
    elif parquet_am is None:
        print(
            "Note: stage5_scored.parquet uses v2 avail_unit tiers; "
            "skipped legacy availability_multiplier validation.",
            file=sys.stderr,
        )

    factors_df = pl.DataFrame(factor_rows)
    n = factors_df.height

    stats_rows: list[dict] = []
    for name in FACTOR_NAMES:
        values = to_float_array(factors_df[name])
        stats = compute_signal_stats(values)
        at_floor = _count_at_floor(values, name)
        at_ceiling = int(np.sum(np.isclose(values, 1.0)))
        flags = _flag_subfactor(
            name,
            stats,
            at_floor,
            at_ceiling,
            missing_counts[name],
            n,
        )
        stats_rows.append(
            {
                "factor_name": name,
                **stats,
                "count_at_floor": at_floor,
                "count_at_ceiling": at_ceiling,
                "count_missing_neutral": missing_counts[name],
                "flags": ";".join(flags),
            }
        )

    stats_df = pl.DataFrame(stats_rows)
    stats_df.write_csv(output_dir / "exp6_avail_subfactors.csv")

    corr_rows: list[dict] = []
    arrays = {name: to_float_array(factors_df[name]) for name in FACTOR_NAMES}
    for sig_a in FACTOR_NAMES:
        row: dict = {"factor": sig_a}
        xa = arrays[sig_a]
        for sig_b in FACTOR_NAMES:
            yb = arrays[sig_b]
            min_len = min(len(xa), len(yb))
            if min_len < 2 or np.std(xa[:min_len]) == 0 or np.std(yb[:min_len]) == 0:
                row[sig_b] = None
            else:
                row[sig_b] = float(spearmanr(xa[:min_len], yb[:min_len])[0])
        corr_rows.append(row)

    corr_df = pl.DataFrame(corr_rows)
    corr_df.write_csv(output_dir / "exp6_avail_subfactor_correlations.csv")

    baseline_mults = []
    for row in factor_rows:
        baseline_mults.append(availability_multiplier(row, cfg))
    baseline_arr = np.array(baseline_mults, dtype=float)
    baseline_var = float(np.var(baseline_arr, ddof=0))

    variance_rows: list[dict] = []
    for neutralized in FACTOR_NAMES:
        neutralized_mults = []
        for factors in factor_rows:
            modified = dict(factors)
            modified[neutralized] = 1.0
            neutralized_mults.append(availability_multiplier(modified, cfg))
        neutral_var = float(np.var(np.array(neutralized_mults, dtype=float), ddof=0))
        contribution = (
            (baseline_var - neutral_var) / baseline_var * 100 if baseline_var > 0 else 0.0
        )
        variance_rows.append(
            {
                "factor_neutralized": neutralized,
                "avail_mult_variance_without_factor": neutral_var,
                "avail_mult_variance_baseline": baseline_var,
                "variance_contribution_pct": round(contribution, 4),
            }
        )

    variance_df = pl.DataFrame(variance_rows)
    variance_df.write_csv(output_dir / "exp6_avail_variance_contribution.csv")

    return {
        "subfactors": stats_df,
        "correlations": corr_df,
        "variance_contribution": variance_df,
        "mismatch_count": mismatch_count,
    }
