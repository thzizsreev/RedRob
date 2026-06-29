"""CSV/JSON report writers for Q1/Q2 vector experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _tier_label(candidate_id: str, tier_map: dict[str, str]) -> str:
    return tier_map.get(candidate_id, "other")


def build_tier_map(candidates_doc: dict[str, Any]) -> dict[str, str]:
    tier_map: dict[str, str] = {}
    for tier_key in (
        "tier_A_keyword_rich",
        "tier_B_outcome_language",
        "tier_C_weak_tail",
        "tier_D_phd_research",
    ):
        for entry in candidates_doc.get(tier_key, []):
            cid = entry if isinstance(entry, str) else entry["id"]
            tier_map[cid] = tier_key
    return tier_map


def write_config_csv(
    path: Path,
    *,
    config_id: int,
    candidate_ids: np.ndarray,
    q1_scores: np.ndarray,
    q2_scores: np.ndarray,
    q1_ranks: np.ndarray,
    q2_ranks: np.ndarray,
    tier_map: dict[str, str],
    baseline_q1_scores: np.ndarray | None = None,
    baseline_q2_scores: np.ndarray | None = None,
    baseline_q1_ranks: np.ndarray | None = None,
    baseline_q2_ranks: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "config_id",
        "q1_score",
        "q2_score",
        "q1_rank",
        "q2_rank",
        "tier_label",
        "baseline_q1_score",
        "baseline_q2_score",
        "baseline_q1_rank",
        "baseline_q2_rank",
        "q1_delta_score",
        "q2_delta_score",
        "q1_delta_rank",
        "q2_delta_rank",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, cid in enumerate(candidate_ids):
            row: dict[str, Any] = {
                "candidate_id": str(cid),
                "config_id": config_id,
                "q1_score": float(q1_scores[i]),
                "q2_score": float(q2_scores[i]),
                "q1_rank": int(q1_ranks[i]),
                "q2_rank": int(q2_ranks[i]),
                "tier_label": _tier_label(str(cid), tier_map),
            }
            if baseline_q1_scores is not None:
                row["baseline_q1_score"] = float(baseline_q1_scores[i])
                row["baseline_q2_score"] = float(baseline_q2_scores[i])  # type: ignore[index]
                row["baseline_q1_rank"] = int(baseline_q1_ranks[i])  # type: ignore[index]
                row["baseline_q2_rank"] = int(baseline_q2_ranks[i])  # type: ignore[index]
                row["q1_delta_score"] = float(q1_scores[i] - baseline_q1_scores[i])
                row["q2_delta_score"] = float(q2_scores[i] - baseline_q2_scores[i])  # type: ignore[index]
                row["q1_delta_rank"] = int(baseline_q1_ranks[i] - q1_ranks[i])  # type: ignore[index]
                row["q2_delta_rank"] = int(baseline_q2_ranks[i] - q2_ranks[i])  # type: ignore[index]
            writer.writerow(row)


def write_facet_breakdown_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_synthetic_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_summary_csv(path, rows)


def write_acceptance_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def print_facet_breakdown(
    candidate_id: str,
    candidate_name: str,
    config_name: str,
    breakdown: list[dict[str, float | str]],
    centroid_score: float,
    baseline_score: float | None,
    rank: int | None,
    baseline_rank: int | None,
) -> None:
    print(f"\nCANDIDATE: {candidate_id} - {candidate_name}")
    print(f"CONFIG: {config_name}")
    print("Facet scores (raw dot product vs candidate vector):")
    for row in breakdown:
        print(
            f"  {row['facet_id']}: {row['raw_dot']:.4f} "
            f"(weight {row['weight']:.2f} -> contribution {row['contribution']:.4f})"
        )
    print(f"Centroid dot product: {centroid_score:.4f}")
    if baseline_score is not None:
        print(f"Baseline dot product: {baseline_score:.4f}")
        print(f"Delta: {centroid_score - baseline_score:+.4f}")
    if rank is not None and baseline_rank is not None:
        print(f"Rank change: {baseline_rank - rank:+d} positions ({baseline_rank} -> {rank})")
