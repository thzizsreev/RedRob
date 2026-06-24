"""Anchor similarity scoring for Stage 1 filtering."""

from __future__ import annotations

import numpy as np


def compute_candidate_similarity(
    candidate_vec: np.ndarray,
    anchor_vec: np.ndarray,
) -> float:
    """Inner product similarity (vectors are pre-normalized at construction)."""
    return float(np.dot(candidate_vec, anchor_vec))
