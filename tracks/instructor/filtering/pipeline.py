"""Stage 1 filtering pipeline: cluster, rank, filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracks.instructor.clustering import assign_cluster_labels, reduce_for_clustering
from tracks.instructor.config import (
    STAGE1_FLOOR,
    STAGE1_RANDOM_SEED,
    UMAP_CLUSTERING_DIMS,
    UMAP_N_NEIGHBORS,
)
from tracks.instructor.filtering.filter import filter_candidates_by_cluster
from tracks.instructor.filtering.rank import rank_clusters_by_anchor_similarity


@dataclass(frozen=True)
class Stage1Result:
    filtered_ids: set[str]
    labels: np.ndarray
    ranked_clusters: list[tuple[int, float, int]]
    n_clusters: int
    noise_count: int
    noise_ratio: float


def run_stage1_filtering(
    candidate_ids: list[str],
    vectors: np.ndarray,
    anchor_vec: np.ndarray,
    *,
    floor: int = STAGE1_FLOOR,
    random_seed: int = STAGE1_RANDOM_SEED,
    clustering_dims: int = UMAP_CLUSTERING_DIMS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    include_noise_as_last_resort: bool = True,
) -> Stage1Result:
    reduced = reduce_for_clustering(
        vectors,
        n_components=clustering_dims,
        random_state=random_seed,
        n_neighbors=n_neighbors,
    )
    labels = assign_cluster_labels(reduced, sample_size=len(candidate_ids))
    ranked = rank_clusters_by_anchor_similarity(
        candidate_ids, vectors, labels, anchor_vec
    )
    filtered_ids = filter_candidates_by_cluster(
        candidate_ids,
        labels,
        ranked,
        floor=floor,
        include_noise_as_last_resort=include_noise_as_last_resort,
    )

    unique_labels = set(int(x) for x in labels)
    n_clusters = len([label for label in unique_labels if label >= 0])
    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / len(labels) if len(labels) else 0.0

    return Stage1Result(
        filtered_ids=filtered_ids,
        labels=labels,
        ranked_clusters=ranked,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
    )
