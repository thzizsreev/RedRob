"""Shared statistics helpers for diagnostic experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import kendalltau, pearsonr, spearmanr

from columns import GROUP_A_SIGNALS, GROUP_B_SIGNALS, GROUP_C_SIGNALS, NORMALIZED_SIGNALS


def to_float_array(series: pl.Series) -> np.ndarray:
    arr = series.cast(pl.Float64, strict=False).to_numpy()
    return arr[~np.isnan(arr)]


def compute_signal_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "iqr": None,
            "cv": None,
        }

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    p5, p25, p50, p75, p95 = [float(x) for x in np.percentile(values, [5, 25, 50, 75, 95])]
    iqr = p75 - p25
    cv = std / mean if mean != 0 else None

    return {
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "range": max_v - min_v,
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "iqr": iqr,
        "cv": cv,
    }


def flag_variance_stats(
    signal_name: str,
    group: str,
    stats: dict[str, float | None],
) -> list[str]:
    flags: list[str] = []
    std = stats.get("std")
    iqr = stats.get("iqr")
    range_v = stats.get("range")
    p25 = stats.get("p25")
    min_v = stats.get("min")

    if std is not None and group in ("A", "B") and std < 0.05:
        flags.append("LOW_VARIANCE")

    if iqr is not None and signal_name in NORMALIZED_SIGNALS and iqr < 0.03:
        flags.append("COMPRESSED")

    if range_v is not None and group == "B" and range_v > 0.5:
        flags.append("WIDE_RANGE")

    if (
        signal_name == "tier3_scaled"
        and p25 is not None
        and min_v is not None
        and p25 == min_v
    ):
        flags.append("FLOOR_DOMINATED")

    return flags


def rank_series(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def compute_rank_deltas(input_vals: np.ndarray, output_vals: np.ndarray) -> dict[str, float | int]:
    input_ranks = rank_series(input_vals)
    output_ranks = rank_series(output_vals)
    deltas = np.abs(output_ranks - input_ranks)
    return {
        "mean_position_delta": float(np.mean(deltas)),
        "max_position_delta": float(np.max(deltas)),
        "candidates_moved_10plus": int(np.sum(deltas > 10)),
        "candidates_moved_20plus": int(np.sum(deltas > 20)),
    }


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(pearsonr(x, y)[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y)[0])


def correlation_pair_flags(
    signal_a: str,
    signal_b: str,
    pearson: float | None,
    spearman: float | None,
) -> list[str]:
    flags: list[str] = []
    if spearman is None:
        return flags

    abs_s = abs(spearman)
    if abs_s > 0.70:
        flags.append("HIGHLY_CORRELATED")
    if signal_a != signal_b:
        if signal_a == "cross_encoder_score" and abs(spearman) > 0.70:
            flags.append("REDUNDANT_WITH_CE")
        elif signal_b == "cross_encoder_score" and abs(spearman) > 0.70:
            flags.append("REDUNDANT_WITH_CE")
    if abs_s < 0.30:
        flags.append("GENUINELY_INDEPENDENT")
    return flags


def write_csv(rows: list[dict[str, Any]], path) -> None:
    if not rows:
        pl.DataFrame(rows).write_csv(path)
        return
    pl.DataFrame(rows).write_csv(path)


def signal_groups() -> list[tuple[str, str]]:
    groups: list[tuple[str, str]] = []
    for sig in GROUP_A_SIGNALS:
        groups.append((sig, "A"))
    for sig in GROUP_B_SIGNALS:
        groups.append((sig, "B"))
    for sig in GROUP_C_SIGNALS:
        groups.append((sig, "C"))
    return groups
