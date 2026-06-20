"""Floor-based cluster filtering for Stage 1."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def filter_candidates_by_cluster(
    candidate_ids: list[str],
    cluster_labels: np.ndarray,
    ranked_clusters: list[tuple[int, float, int]],
    *,
    floor: int = 100,
    include_noise_as_last_resort: bool = True,
) -> set[str]:
    """
    Walk ranked clusters atomically until cumulative count >= floor.

    Noise (label -1) is skipped during the normal walk. When
    include_noise_as_last_resort is True and real clusters are exhausted
    without reaching floor, all noise points are added as a last resort.
    """
    label_to_ids: dict[int, list[str]] = defaultdict(list)
    for cid, label in zip(candidate_ids, cluster_labels):
        label_to_ids[int(label)].append(cid)

    filtered_set: set[str] = set()

    for label, _median_sim, _size in ranked_clusters:
        if label == -1:
            continue
        filtered_set.update(label_to_ids[label])
        if len(filtered_set) >= floor:
            break

    if len(filtered_set) < floor and include_noise_as_last_resort:
        filtered_set.update(label_to_ids.get(-1, []))

    return filtered_set
