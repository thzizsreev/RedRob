"""Rank clusters by median anchor similarity."""

from __future__ import annotations

from collections import defaultdict
import statistics

import numpy as np


def compute_anchor_similarities(
    vectors: np.ndarray,
    anchor_vec: np.ndarray,
) -> np.ndarray:
    return vectors @ anchor_vec


def rank_clusters_from_similarities(
    cluster_labels: np.ndarray,
    anchor_similarities: np.ndarray,
) -> list[tuple[int, float, int]]:
    """
    Return (cluster_label, median_similarity, cluster_size) sorted by
    descending median similarity, tie-broken by ascending label.
    """
    cluster_to_sims: dict[int, list[float]] = defaultdict(list)
    for label, sim in zip(cluster_labels, anchor_similarities):
        cluster_to_sims[int(label)].append(float(sim))

    results: list[tuple[int, float, int]] = []
    for label, sims in cluster_to_sims.items():
        results.append((label, statistics.median(sims), len(sims)))

    results.sort(key=lambda x: (-x[1], x[0]))
    return results
