"""HDBSCAN cluster assignment on UMAP-reduced vectors."""

from __future__ import annotations

import hdbscan
import numpy as np


def min_cluster_size(sample_size: int) -> int:
    return max(15, int(0.015 * sample_size))


def assign_cluster_labels(
    reduced_vectors: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """Assign HDBSCAN cluster labels; -1 denotes noise."""
    mcs = min_cluster_size(sample_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=mcs,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced_vectors)
