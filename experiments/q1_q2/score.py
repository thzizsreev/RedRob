"""Dot-product scoring and ranking for Q1/Q2 vector experiments."""

from __future__ import annotations

import numpy as np


def score_all_candidates(query_vec: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    """Raw dot product scores for all candidates."""
    return candidate_matrix @ query_vec.astype(np.float32)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score (descending order)."""
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def lookup_id_index(candidate_ids: np.ndarray, target_id: str) -> int | None:
    matches = np.where(candidate_ids == target_id)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def score_at_id(
    scores: np.ndarray,
    ranks: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_id: str,
) -> tuple[float | None, int | None]:
    idx = lookup_id_index(candidate_ids, candidate_id)
    if idx is None:
        return None, None
    return float(scores[idx]), int(ranks[idx])
