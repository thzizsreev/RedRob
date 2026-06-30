"""Quantitative clustering metrics for K-means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    n_clusters: int
    cluster_sizes: dict[int, int]
    silhouette: float | None
    inertia: float


def compute_metrics(
    clustering_coords: np.ndarray,
    labels: np.ndarray,
    inertia: float,
) -> ClusteringResult:
    unique_labels = sorted({int(x) for x in labels})
    n_clusters = len(unique_labels)

    cluster_sizes: dict[int, int] = {
        label: int(np.sum(labels == label)) for label in unique_labels
    }

    silhouette: float | None = None
    if n_clusters >= 2 and len(labels) > n_clusters:
        try:
            silhouette = float(
                silhouette_score(
                    clustering_coords,
                    labels,
                    metric="euclidean",
                )
            )
        except ValueError:
            silhouette = None

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        silhouette=silhouette,
        inertia=inertia,
    )
