"""Vector steering math from plan_1_resume_base_steering.md."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from constants import DIMENSIONS


def interpolate(v_lo: np.ndarray, v_hi: np.ndarray, score: float) -> np.ndarray:
    return (1.0 - score) * v_lo + score * v_hi


def blend_candidate(v_candidate: np.ndarray, v_steer: np.ndarray, beta: float) -> np.ndarray:
    return beta * v_candidate + (1.0 - beta) * v_steer


def compute_final_vectors(
    v_candidate: np.ndarray,
    anchors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    scores: Mapping[str, float],
    beta: float,
) -> dict[str, np.ndarray]:
    """Return v_final for each dimension."""
    final: dict[str, np.ndarray] = {}
    for dim in DIMENSIONS:
        v_lo, v_hi = anchors[dim]
        v_steer = interpolate(v_lo, v_hi, scores[dim])
        final[dim] = blend_candidate(v_candidate, v_steer, beta)
    return final
