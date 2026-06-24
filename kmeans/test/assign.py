"""K-means cluster assignment."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_candidates_kmeans(
    reduced_vectors: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Return cluster labels and inertia from K-means fit."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(reduced_vectors)
    return labels.astype(np.int32), float(kmeans.inertia_)
