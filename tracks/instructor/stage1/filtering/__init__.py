"""Stage 1 cluster-based filtering."""

from tracks.instructor.stage1.filtering.filter import filter_candidates_by_cluster
from tracks.instructor.stage1.filtering.pipeline import (
    Stage1Result,
    cluster_candidates,
    filter_from_labels,
    run_stage1_filtering,
)
from tracks.instructor.stage1.filtering.rank import (
    compute_anchor_similarities,
    rank_clusters_by_anchor_similarity,
    rank_clusters_from_similarities,
)
from tracks.instructor.stage1.filtering.similarity import compute_candidate_similarity

__all__ = [
    "Stage1Result",
    "cluster_candidates",
    "compute_anchor_similarities",
    "compute_candidate_similarity",
    "filter_candidates_by_cluster",
    "filter_from_labels",
    "rank_clusters_by_anchor_similarity",
    "rank_clusters_from_similarities",
    "run_stage1_filtering",
]
