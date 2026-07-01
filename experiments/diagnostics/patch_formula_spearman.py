#!/usr/bin/env python3
"""Patch spearman correlation keys in formula_design_metrics from exp2 matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DIAG_ROOT = Path(__file__).resolve().parent
RESULTS_JSON = _DIAG_ROOT / "output" / "diagnostics_results.json"

SPEARMAN_PAIRS: list[tuple[str, str, str]] = [
    ("spearman(cross_encoder_score, q1_score)", "cross_encoder_score", "q1_score"),
    ("spearman(cross_encoder_score, fused_score)", "cross_encoder_score", "fused_score"),
    ("spearman(cross_encoder_score, q2_score)", "cross_encoder_score", "q2_score"),
    ("spearman(cross_encoder_score, skill_score)", "cross_encoder_score", "skill_score"),
    ("spearman(q1_score, q2_score)", "q1_score", "q2_score"),
]


def extract_spearman(matrix: dict, row_key: str, col_key: str) -> float | None:
    if row_key not in matrix:
        raise KeyError(
            f"Missing row key '{row_key}' in experiments.exp2_correlation.spearman_matrix"
        )
    row = matrix[row_key]
    if col_key not in row:
        raise KeyError(
            f"Missing column key '{col_key}' for row '{row_key}' "
            f"in experiments.exp2_correlation.spearman_matrix"
        )
    value = row[col_key]
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            f"Expected float or null at spearman_matrix['{row_key}']['{col_key}'], "
            f"got bool: {value!r}"
        )
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(
        f"Expected float or null at spearman_matrix['{row_key}']['{col_key}'], "
        f"got {type(value).__name__}: {value!r}"
    )


def patch_formula_spearman(path: Path) -> dict[str, float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    matrix = data["experiments"]["exp2_correlation"]["spearman_matrix"]

    if "formula_design_metrics" not in data:
        raise KeyError("Missing top-level key 'formula_design_metrics'")

    updated: dict[str, float | None] = {}
    for metric_key, row_key, col_key in SPEARMAN_PAIRS:
        updated[metric_key] = extract_spearman(matrix, row_key, col_key)
        data["formula_design_metrics"][metric_key] = updated[metric_key]

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return updated


def main() -> None:
    path = RESULTS_JSON
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    updated = patch_formula_spearman(path)
    print(f"Patched {len(updated)} spearman keys in {path}")
    for key, value in updated.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
