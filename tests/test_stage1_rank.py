"""Unit tests for vectorized Stage 1 rank."""

from __future__ import annotations

import statistics

import numpy as np

from tracks.instructor.filtering.rank import (
    compute_anchor_similarities,
    rank_clusters_by_anchor_similarity,
    rank_clusters_from_similarities,
)
from tracks.instructor.filtering.similarity import compute_candidate_similarity


def test_vectorized_rank_matches_loop() -> None:
    rng = np.random.default_rng(0)
    n, dim = 40, 16
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    anchor = rng.standard_normal(dim).astype(np.float32)
    labels = np.array([i % 5 if i < 35 else -1 for i in range(n)], dtype=np.int32)

    loop_ranked = rank_clusters_by_anchor_similarity(
        [f"C{i}" for i in range(n)], vectors, labels, anchor
    )
    sims = compute_anchor_similarities(vectors, anchor)
    vec_ranked = rank_clusters_from_similarities(labels, sims)

    assert loop_ranked == vec_ranked

    for i in range(n):
        expected = compute_candidate_similarity(vectors[i], anchor)
        assert abs(float(sims[i]) - expected) < 1e-5

    cluster_to_sims: dict[int, list[float]] = {}
    for label, sim in zip(labels, sims):
        cluster_to_sims.setdefault(int(label), []).append(float(sim))
    for label, median_sim, size in vec_ranked:
        assert size == len(cluster_to_sims[label])
        assert median_sim == statistics.median(cluster_to_sims[label])
