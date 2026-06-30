"""Floor-based cluster filtering for K-means."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def filter_candidates_by_cluster(
    candidate_ids: list[str],
    cluster_labels: np.ndarray,
    ranked_clusters: list[tuple[int, float, int]],
    *,
    floor: int = 100,
) -> set[str]:
    """Walk ranked clusters atomically until cumulative count >= floor."""
    label_to_ids: dict[int, list[str]] = defaultdict(list)
    for cid, label in zip(candidate_ids, cluster_labels):
        label_to_ids[int(label)].append(cid)

    filtered_set: set[str] = set()
    for label, _median_sim, _size in ranked_clusters:
        filtered_set.update(label_to_ids[label])
        if len(filtered_set) >= floor:
            break

    return filtered_set
