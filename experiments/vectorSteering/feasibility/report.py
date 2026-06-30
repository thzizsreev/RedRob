"""CSV and JSON report writers for vector steering tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PHASE_A_FIELDS = [
    "phase",
    "input_id",
    "s_value",
    "run_number",
    "original_text",
    "decoded_text",
    "deterministic_auto",
    "human_fidelity",
    "human_coherent",
    "human_sentiment",
]

PHASE_B_FIELDS = PHASE_A_FIELDS


def write_results_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
