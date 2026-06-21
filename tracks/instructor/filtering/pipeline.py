"""Stage 1 filtering pipeline: cluster, rank, filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracks.instructor.clustering import assign_cluster_labels, min_cluster_size, reduce_for_clustering
from tracks.instructor.config import (
    STAGE1_FLOOR,
    STAGE1_HDBSCAN_CORE_DIST_N_JOBS,
    STAGE1_RANDOM_SEED,
    STAGE1_UMAP_N_JOBS,
    UMAP_CLUSTERING_DIMS,
    UMAP_N_NEIGHBORS,
)
from tracks.instructor.filtering.filter import filter_candidates_by_cluster
from tracks.instructor.filtering.rank import (
    compute_anchor_similarities,
    rank_clusters_from_similarities,
)


@dataclass(frozen=True)
class Stage1Result:
    filtered_ids: set[str]
    labels: np.ndarray
    ranked_clusters: list[tuple[int, float, int]]
    n_clusters: int
    noise_count: int
    noise_ratio: float
    anchor_similarities: np.ndarray | None = None


def _cluster_stats(labels: np.ndarray) -> tuple[int, int, float]:
    unique_labels = set(int(x) for x in labels)
    n_clusters = len([label for label in unique_labels if label >= 0])
    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / len(labels) if len(labels) else 0.0
    return n_clusters, noise_count, noise_ratio


def cluster_candidates(
    vectors: np.ndarray,
    *,
    random_seed: int = STAGE1_RANDOM_SEED,
    clustering_dims: int = UMAP_CLUSTERING_DIMS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    umap_n_jobs: int = STAGE1_UMAP_N_JOBS,
    hdbscan_core_dist_n_jobs: int = STAGE1_HDBSCAN_CORE_DIST_N_JOBS,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase A: UMAP reduce + HDBSCAN cluster assignment."""
    reduced = reduce_for_clustering(
        vectors,
        n_components=clustering_dims,
        random_state=random_seed,
        n_neighbors=n_neighbors,
        n_jobs=umap_n_jobs,
    )
    labels = assign_cluster_labels(
        reduced,
        sample_size=len(vectors),
        core_dist_n_jobs=hdbscan_core_dist_n_jobs,
    )
    return reduced, labels


def filter_from_labels(
    candidate_ids: list[str],
    vectors: np.ndarray,
    labels: np.ndarray,
    anchor_vec: np.ndarray,
    *,
    floor: int = STAGE1_FLOOR,
    include_noise_as_last_resort: bool = True,
    anchor_similarities: np.ndarray | None = None,
) -> Stage1Result:
    """Phase B: rank clusters by anchor similarity and filter to floor."""
    if anchor_similarities is None:
        anchor_similarities = compute_anchor_similarities(vectors, anchor_vec)

    ranked = rank_clusters_from_similarities(labels, anchor_similarities)
    filtered_ids = filter_candidates_by_cluster(
        candidate_ids,
        labels,
        ranked,
        floor=floor,
        include_noise_as_last_resort=include_noise_as_last_resort,
    )
    n_clusters, noise_count, noise_ratio = _cluster_stats(labels)

    return Stage1Result(
        filtered_ids=filtered_ids,
        labels=labels,
        ranked_clusters=ranked,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
        anchor_similarities=anchor_similarities,
    )


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
    umap_n_jobs: int = STAGE1_UMAP_N_JOBS,
    hdbscan_core_dist_n_jobs: int = STAGE1_HDBSCAN_CORE_DIST_N_JOBS,
) -> Stage1Result:
    """Run phase A + B sequentially (legacy convenience wrapper)."""
    reduced, labels = cluster_candidates(
        vectors,
        random_seed=random_seed,
        clustering_dims=clustering_dims,
        n_neighbors=n_neighbors,
        umap_n_jobs=umap_n_jobs,
        hdbscan_core_dist_n_jobs=hdbscan_core_dist_n_jobs,
    )
    del reduced
    return filter_from_labels(
        candidate_ids,
        vectors,
        labels,
        anchor_vec,
        floor=floor,
        include_noise_as_last_resort=include_noise_as_last_resort,
    )


def min_cluster_size_for_sample(sample_size: int) -> int:
    return min_cluster_size(sample_size)
