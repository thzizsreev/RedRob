"""Rank clusters by median anchor similarity."""

from __future__ import annotations

from collections import defaultdict
import statistics

import numpy as np

from tracks.instructor.filtering.similarity import compute_candidate_similarity


def rank_clusters_by_anchor_similarity(
    candidate_ids: list[str],
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
    anchor_vec: np.ndarray,
) -> list[tuple[int, float, int]]:
    """
    Return (cluster_label, median_similarity, cluster_size) sorted by
    descending median similarity, tie-broken by ascending label.
    """
    del candidate_ids  # labels and vectors are aligned by index

    cluster_to_sims: dict[int, list[float]] = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        sim = compute_candidate_similarity(vectors[i], anchor_vec)
        cluster_to_sims[int(label)].append(sim)

    results: list[tuple[int, float, int]] = []
    for label, sims in cluster_to_sims.items():
        results.append((label, statistics.median(sims), len(sims)))

    results.sort(key=lambda x: (-x[1], x[0]))
    return results
