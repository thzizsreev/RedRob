"""Min-max normalization helpers for Stage 5."""

from __future__ import annotations

import numpy as np


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi == lo:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)
