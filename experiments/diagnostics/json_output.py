"""JSON output for Stage 5 diagnostics — structured for humans and AI agents."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


def _split_flags(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in str(value).split(";") if part]


def _row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in row.items():
        if key in ("flags", "flag"):
            out["flags"] = _split_flags(val)
        elif isinstance(val, float):
            out[key] = round(val, 6) if val == val else None
        else:
            out[key] = val
    return out


def df_to_records(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [_row_to_json(dict(row)) for row in df.iter_rows(named=True)]


def matrix_df_to_dict(df: pl.DataFrame, index_col: str = "signal") -> dict[str, dict[str, float | None]]:
    matrix: dict[str, dict[str, float | None]] = {}
    for row in df.iter_rows(named=True):
        row_key = str(row[index_col])
        matrix[row_key] = {}
        for col, val in row.items():
            if col == index_col:
                continue
            if val is None:
                matrix[row_key][col] = None
            else:
                matrix[row_key][col] = round(float(val), 6)
    return matrix


def _parse_metric_value(value: str) -> int | float | str | None:
    if value in ("N/A", "—", ""):
        return None
    if value.endswith("%"):
        try:
            return round(float(value[:-1]), 4)
        except ValueError:
            return value
    try:
        if "." in value:
            return round(float(value), 6)
        return int(value)
    except ValueError:
        return value


def _build_summary_narrative(
    exp1: pl.DataFrame,
    exp2: dict,
    exp3: pl.DataFrame,
    exp4: dict,
    exp5: pl.DataFrame,
    exp6: dict,
) -> dict[str, Any]:
    sorted_exp1 = exp1.sort("std", descending=True, nulls_last=True)
    flat = sorted_exp1.filter(pl.col("flags").str.contains("LOW_VARIANCE"))
    discriminating = sorted_exp1.filter(
        ~pl.col("flags").str.contains("LOW_VARIANCE") & pl.col("std").is_not_null()
    )
    high_corr = exp2["high_correlation_pairs"].filter(
        pl.col("flag").str.contains("HIGHLY_CORRELATED")
    )
    exp3_sorted = exp3.sort("spearman", nulls_last=False)
    dominant = exp3_sorted.filter(pl.col("flags").str.contains("DOMINANT_LAYER"))
    summary = exp4["summary_dict"]
    subfactors = exp6["subfactors"]

    keep = subfactors.filter(
        ~pl.col("flags").str.contains("NEAR_CONSTANT")
        & ~pl.col("flags").str.contains("MISSING_DOMINATED")
    )
    simplify = subfactors.filter(pl.col("flags").str.contains("FLOOR_HEAVY"))
    remove = subfactors.filter(pl.col("flags").str.contains("NEAR_CONSTANT"))

    section1_text = (
        f"Signals with real discrimination power include: "
        f"{', '.join(discriminating['signal_name'].to_list()[:8]) or 'none'}. "
        f"Nearly flat signals (LOW_VARIANCE) include: "
        f"{', '.join(flat['signal_name'].to_list()[:8]) or 'none'}."
    )

    redundancy_notes = []
    for row in high_corr.iter_rows(named=True):
        redundancy_notes.append(
            f"{row['signal_a']} and {row['signal_b']} measure nearly the same thing "
            f"(spearman={row['spearman']:.4f}); using both adds minimal new information."
        )

    layer_notes = []
    for row in exp3_sorted.iter_rows(named=True):
        sp = row["spearman"]
        sp_s = f"{sp:.4f}" if sp is not None else "N/A"
        layer_notes.append(
            f"{row['transition']}: spearman={sp_s}, "
            f"{row['candidates_moved_20plus']} candidates moved 20+ positions."
        )

    dominant_text = None
    if not dominant.is_empty():
        d = dominant.row(0, named=True)
        dominant_text = (
            f"{d['transition']} is the most disruptive layer, reshuffling "
            f"{d['candidates_moved_20plus']} candidates by more than 20 positions."
        )

    availability_text = (
        "Wrong flips occur when a technically stronger candidate (higher score_after_t2) ranks below "
        "a weaker one after tier3 availability is applied. Large-gap flips and top-100 displacement "
        "quantify submission-quality cost."
    )

    flagged_exp5 = exp5.filter(
        pl.col("flags").str.contains("RARELY_TRIGGERED")
        | pl.col("flags").str.contains("ALWAYS_TRIGGERED")
    )
    coverage_notes = [
        f"{row['signal_name']} {row['condition']}: {row['pct_triggered']}% ({row['flags']})"
        for row in flagged_exp5.iter_rows(named=True)
    ]

    return {
        "section1_signal_discrimination": {
            "interpretation": section1_text,
            "top_discriminating_signals": discriminating["signal_name"].to_list()[:10],
            "flat_signals": flat["signal_name"].to_list(),
        },
        "section2_redundancy": {
            "highly_correlated_pairs": df_to_records(high_corr),
            "notes": redundancy_notes or ["No pairs flagged HIGHLY_CORRELATED (|spearman| > 0.70)."],
        },
        "section3_layer_disruption": {
            "layers_most_disruptive_first": df_to_records(exp3_sorted),
            "dominant_layer_note": dominant_text,
            "notes": layer_notes,
        },
        "section4_availability_flips": {
            "interpretation": availability_text,
            "metrics": summary,
        },
        "section5_boolean_coverage": {
            "flagged_conditions": coverage_notes or ["No RARELY_TRIGGERED or ALWAYS_TRIGGERED flags."],
        },
        "section6_availability_subfactors": {
            "recommendations": {
                "keep": keep["factor_name"].to_list(),
                "simplify": simplify["factor_name"].to_list(),
                "remove_candidate": remove["factor_name"].to_list(),
            },
        },
    }


def build_diagnostics_json(
    *,
    scored_path: Path,
    jsonl_path: Path,
    candidate_count: int,
    current_date: date,
    exp1: pl.DataFrame,
    exp2: dict,
    exp3: pl.DataFrame,
    exp4: dict,
    exp5: pl.DataFrame,
    exp6: dict,
    section7: dict[str, str],
) -> dict[str, Any]:
    exp1_sorted = exp1.sort("std", descending=True, nulls_last=True)
    formula_metrics = {
        key.replace(" ", "_"): _parse_metric_value(str(val))
        for key, val in section7.items()
    }

    return {
        "meta": {
            "report_type": "stage5_diagnostics",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_count": candidate_count,
            "inputs": {
                "scored_parquet": str(scored_path),
                "candidates_jsonl": str(jsonl_path),
                "current_date": current_date.isoformat(),
            },
        },
        "summary": _build_summary_narrative(exp1, exp2, exp3, exp4, exp5, exp6),
        "experiments": {
            "exp1_signal_variance": {
                "description": "Per-signal variance statistics across Groups A, B, and C.",
                "signals": df_to_records(exp1),
                "signals_sorted_by_std": df_to_records(exp1_sorted),
            },
            "exp2_correlation": {
                "description": "Pairwise Pearson and Spearman correlations; filtered high-correlation pairs.",
                "pearson_matrix": matrix_df_to_dict(exp2["pearson_matrix"], "signal"),
                "spearman_matrix": matrix_df_to_dict(exp2["spearman_matrix"], "signal"),
                "high_correlation_pairs": df_to_records(exp2["high_correlation_pairs"]),
            },
            "exp3_layer_rank_stability": {
                "description": "Rank stability between adjacent scoring layers.",
                "transitions": df_to_records(exp3),
            },
            "exp4_availability_flips": {
                "description": "Pairs where tier3 availability reverses pre-availability ordering.",
                "summary": exp4["summary_dict"],
                "large_gap_flip_details": df_to_records(exp4["detail"]),
            },
            "exp5_boolean_coverage": {
                "description": "Trigger rates for boolean, categorical, penalty, and bonus conditions.",
                "coverage": df_to_records(exp5),
            },
            "exp6_availability_subfactors": {
                "description": "Decomposition of the seven availability sub-factors.",
                "subfactors": df_to_records(exp6["subfactors"]),
                "subfactor_correlations": matrix_df_to_dict(
                    exp6["correlations"], "factor"
                ),
                "variance_contribution": df_to_records(exp6["variance_contribution"]),
                "availability_multiplier_mismatches": exp6.get("mismatch_count", 0),
            },
        },
        "formula_design_metrics": formula_metrics,
    }


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_json_outputs(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    consolidated_path = output_dir / "diagnostics_results.json"
    write_json_file(consolidated_path, payload)

    experiments = payload["experiments"]
    write_json_file(output_dir / "exp1_signal_variance.json", experiments["exp1_signal_variance"])
    write_json_file(output_dir / "exp2_correlation.json", experiments["exp2_correlation"])
    write_json_file(output_dir / "exp3_layer_rank_stability.json", experiments["exp3_layer_rank_stability"])
    write_json_file(output_dir / "exp4_availability_flips.json", experiments["exp4_availability_flips"])
    write_json_file(output_dir / "exp5_boolean_coverage.json", experiments["exp5_boolean_coverage"])
    write_json_file(output_dir / "exp6_availability_subfactors.json", experiments["exp6_availability_subfactors"])

    return consolidated_path
