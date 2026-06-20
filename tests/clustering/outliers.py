"""Stage 8 — export noise-labeled candidates for manual outlier review."""

from __future__ import annotations

import numpy as np


def _profile_summary(record: dict) -> dict:
    profile = record.get("profile", {})
    return {
        "candidate_id": record["candidate_id"],
        "current_title": profile.get("current_title", ""),
        "years_of_experience": profile.get("years_of_experience"),
        "headline": profile.get("headline", ""),
        "summary_excerpt": str(profile.get("summary", ""))[:240],
    }


def build_noise_export(
    candidate_ids: list[str],
    records: list[dict],
    labels: np.ndarray,
) -> dict:
    noise_indices = [i for i, label in enumerate(labels) if int(label) == -1]
    noise_candidates = [_profile_summary(records[i]) for i in noise_indices]
    total = len(labels)
    return {
        "noise_count": len(noise_candidates),
        "noise_ratio": len(noise_candidates) / total if total else 0.0,
        "instructions": (
            "Cross-check noise candidates against honeypot/heuristic signals manually."
        ),
        "candidates": noise_candidates,
    }
