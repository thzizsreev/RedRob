"""Stage 1 cluster-based filtering."""

from tracks.instructor.filtering.filter import filter_candidates_by_cluster
from tracks.instructor.filtering.pipeline import Stage1Result, run_stage1_filtering
from tracks.instructor.filtering.rank import rank_clusters_by_anchor_similarity
from tracks.instructor.filtering.similarity import compute_candidate_similarity

__all__ = [
    "Stage1Result",
    "compute_candidate_similarity",
    "filter_candidates_by_cluster",
    "rank_clusters_by_anchor_similarity",
    "run_stage1_filtering",
]
