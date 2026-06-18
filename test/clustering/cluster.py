"""Stage 3–4 — HDBSCAN clustering and quantitative metrics."""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    min_cluster_size: int
    min_samples: int
    n_clusters: int
    noise_count: int
    noise_ratio: float
    silhouette: float | None
    cluster_sizes: dict[int, int]


def _min_cluster_size(sample_size: int) -> int:
    return max(15, int(0.015 * sample_size))


def run_hdbscan(
    clustering_coords: np.ndarray,
    sample_size: int,
) -> tuple[np.ndarray, int, int]:
    min_cluster_size = _min_cluster_size(sample_size)
    min_samples = min_cluster_size
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(clustering_coords)
    return labels, min_cluster_size, min_samples


def compute_metrics(
    clustering_coords: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> ClusteringResult:
    unique_labels = sorted(set(int(x) for x in labels))
    n_clusters = len([label for label in unique_labels if label >= 0])
    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / len(labels) if len(labels) else 0.0

    cluster_sizes: dict[int, int] = {}
    for label in unique_labels:
        if label >= 0:
            cluster_sizes[label] = int(np.sum(labels == label))

    silhouette: float | None = None
    non_noise = labels >= 0
    if n_clusters >= 2 and int(non_noise.sum()) > n_clusters:
        try:
            silhouette = float(
                silhouette_score(
                    clustering_coords[non_noise],
                    labels[non_noise],
                    metric="euclidean",
                )
            )
        except ValueError:
            silhouette = None

    return ClusteringResult(
        labels=labels,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
        silhouette=silhouette,
        cluster_sizes=cluster_sizes,
    )
