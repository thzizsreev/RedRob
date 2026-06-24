"""K-means filter pipeline: rank clusters and apply floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kmeans.filter.filter import filter_candidates_by_cluster
from kmeans.filter.rank import compute_anchor_similarities, rank_clusters_from_similarities


@dataclass(frozen=True)
class KMeansFilterResult:
    filtered_ids: set[str]
    labels: np.ndarray
    ranked_clusters: list[tuple[int, float, int]]
    n_clusters: int
    anchor_similarities: np.ndarray


def filter_from_labels(
    candidate_ids: list[str],
    vectors: np.ndarray,
    labels: np.ndarray,
    anchor_vec: np.ndarray,
    *,
    floor: int = 100,
    anchor_similarities: np.ndarray | None = None,
) -> KMeansFilterResult:
    if anchor_similarities is None:
        anchor_similarities = compute_anchor_similarities(vectors, anchor_vec)

    ranked = rank_clusters_from_similarities(labels, anchor_similarities)
    filtered_ids = filter_candidates_by_cluster(
        candidate_ids,
        labels,
        ranked,
        floor=floor,
    )
    n_clusters = len({int(x) for x in labels})

    return KMeansFilterResult(
        filtered_ids=filtered_ids,
        labels=labels,
        ranked_clusters=ranked,
        n_clusters=n_clusters,
        anchor_similarities=anchor_similarities,
    )
