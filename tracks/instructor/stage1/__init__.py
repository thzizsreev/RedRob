"""Stage 1 — HDBSCAN cluster precompute and JD-anchor cluster filtering."""

from tracks.instructor.stage1.artifacts import (
    ClusterManifest,
    Stage1ClusterArtifacts,
    Stage1ArtifactsMissingError,
    Stage1ClusterArtifactsExistError,
    assert_cluster_artifacts_absent,
    cluster_artifacts_exist,
    require_cluster_artifacts,
    save_cluster_artifacts,
    stage1_dir,
    validate_manifest_params,
)
from tracks.instructor.stage1.pipeline import (
    Stage1RunResult,
    build_filtered_output,
    precompute_stage1_clustering,
    print_stage1_summary,
    run_stage1_filter,
    run_stage1_from_artifacts,
    write_stage1_artifacts,
)

__all__ = [
    "ClusterManifest",
    "Stage1ArtifactsMissingError",
    "Stage1ClusterArtifacts",
    "Stage1ClusterArtifactsExistError",
    "Stage1RunResult",
    "assert_cluster_artifacts_absent",
    "build_filtered_output",
    "cluster_artifacts_exist",
    "precompute_stage1_clustering",
    "print_stage1_summary",
    "require_cluster_artifacts",
    "run_stage1_filter",
    "run_stage1_from_artifacts",
    "save_cluster_artifacts",
    "stage1_dir",
    "validate_manifest_params",
    "write_stage1_artifacts",
]
