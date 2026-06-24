"""Stage 0 — cluster precompute wrapper (UMAP + HDBSCAN)."""

from __future__ import annotations

from pathlib import Path

from tracks.instructor.stage1.artifacts import Stage1ClusterArtifacts
from tracks.instructor.stage1.pipeline import precompute_stage1_clustering


def run_cluster_precompute(
    stage0_path: Path,
    stage1_path: Path,
    *,
    random_seed: int,
    overwrite: bool = False,
) -> Stage1ClusterArtifacts:
    return precompute_stage1_clustering(
        stage0_path,
        stage1_path=stage1_path,
        random_seed=random_seed,
        overwrite=overwrite,
    )
