"""Experiment 3 — layer-by-layer rank stability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import kendalltau, spearmanr

from columns import LAYER_TRANSITIONS
from stats_utils import compute_rank_deltas, to_float_array


def _layer_flags(spearman: float | None, is_dominant: bool) -> str:
    flags: list[str] = []
    if spearman is None:
        return ""
    if spearman < 0.80:
        flags.append("HIGH_DISRUPTION")
    if is_dominant:
        flags.append("DOMINANT_LAYER")
    if spearman > 0.97:
        flags.append("NEAR_NOOP")
    return ";".join(flags)


def run_exp3(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    rows: list[dict] = []
    spearman_values: list[tuple[str, float | None]] = []

    for transition, input_col, output_col in LAYER_TRANSITIONS:
        x = to_float_array(df[input_col])
        y = to_float_array(df[output_col])
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            spearman = None
            kendall = None
        else:
            spearman = float(spearmanr(x, y)[0])
            kendall = float(kendalltau(x, y)[0])

        deltas = compute_rank_deltas(x, y)
        spearman_values.append((transition, spearman))
        rows.append(
            {
                "transition": transition,
                "input_col": input_col,
                "output_col": output_col,
                "spearman": spearman,
                "kendall_tau": kendall,
                **deltas,
                "flags": "",
            }
        )

    valid_spearman = [(t, s) for t, s in spearman_values if s is not None]
    dominant_transition = None
    if valid_spearman:
        dominant_transition = min(valid_spearman, key=lambda item: item[1])[0]

    for row in rows:
        row["flags"] = _layer_flags(
            row["spearman"],
            row["transition"] == dominant_transition,
        )

    result = pl.DataFrame(rows)
    result.write_csv(output_dir / "exp3_layer_rank_stability.csv")
    return result
