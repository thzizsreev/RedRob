"""Score-scaled steering in embedding space."""

from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def compute_steering_direction(v_good: np.ndarray, v_bad: np.ndarray) -> np.ndarray:
    return (v_good - v_bad).astype(np.float32)


def steer_vector(v_base: np.ndarray, v_steer: np.ndarray, s: float) -> np.ndarray:
    """Displace v_base by S * v_steer and L2-normalize onto the unit hypersphere."""
    displaced = v_base + (s * v_steer)
    return l2_normalize(displaced)
