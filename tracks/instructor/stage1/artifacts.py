"""Explicit load/save for Stage 1 phase-A cluster precompute artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracks.instructor.core.config import (
    STAGE1_CANDIDATE_VECTORS_FILENAME,
    STAGE1_CLUSTER_LABELS_FILENAME,
    STAGE1_CLUSTER_MANIFEST_FILENAME,
    STAGE1_DIRNAME,
    STAGE1_UMAP_REDUCED_FILENAME,
)


class Stage1ArtifactsMissingError(FileNotFoundError):
    """Raised when phase-B filter is run before phase-A cluster precompute."""


class Stage1ClusterArtifactsExistError(FileExistsError):
    """Raised when phase-A artifacts already exist and overwrite=False."""


@dataclass(frozen=True)
class ClusterManifest:
    n_candidates: int
    vector_dim: int
    random_seed: int
    clustering_dims: int
    n_neighbors: int
    umap_n_jobs: int
    hdbscan_core_dist_n_jobs: int
    min_cluster_size: int

    def to_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "vector_dim": self.vector_dim,
            "random_seed": self.random_seed,
            "clustering_dims": self.clustering_dims,
            "n_neighbors": self.n_neighbors,
            "umap_n_jobs": self.umap_n_jobs,
            "hdbscan_core_dist_n_jobs": self.hdbscan_core_dist_n_jobs,
            "min_cluster_size": self.min_cluster_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClusterManifest:
        return cls(
            n_candidates=int(data["n_candidates"]),
            vector_dim=int(data["vector_dim"]),
            random_seed=int(data["random_seed"]),
            clustering_dims=int(data["clustering_dims"]),
            n_neighbors=int(data["n_neighbors"]),
            umap_n_jobs=int(data["umap_n_jobs"]),
            hdbscan_core_dist_n_jobs=int(data["hdbscan_core_dist_n_jobs"]),
            min_cluster_size=int(data["min_cluster_size"]),
        )


@dataclass(frozen=True)
class Stage1ClusterArtifacts:
    stage1_dir: Path
    candidate_ids: list[str]
    vectors: np.ndarray
    labels: np.ndarray
    reduced: np.ndarray
    manifest: ClusterManifest
    n_clusters: int
    noise_count: int
    noise_ratio: float


def stage1_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / STAGE1_DIRNAME


def _artifact_paths(stage1_path: Path) -> dict[str, Path]:
    return {
        "vectors": stage1_path / STAGE1_CANDIDATE_VECTORS_FILENAME,
        "labels": stage1_path / STAGE1_CLUSTER_LABELS_FILENAME,
        "reduced": stage1_path / STAGE1_UMAP_REDUCED_FILENAME,
        "manifest": stage1_path / STAGE1_CLUSTER_MANIFEST_FILENAME,
    }


def cluster_artifacts_exist(stage1_path: Path) -> bool:
    paths = _artifact_paths(stage1_path)
    return all(path.exists() for path in paths.values())


def _missing_artifact_names(stage1_path: Path) -> list[str]:
    return [
        name for name, path in _artifact_paths(stage1_path).items() if not path.exists()
    ]


def assert_cluster_artifacts_absent(stage1_path: Path, *, overwrite: bool) -> None:
    if cluster_artifacts_exist(stage1_path) and not overwrite:
        raise Stage1ClusterArtifactsExistError(
            f"Stage 1 cluster artifacts already exist in {stage1_path}. "
            "Pass overwrite=True or delete the stage1/ directory before re-running "
            "precompute_stage1_clustering()."
        )


def save_cluster_artifacts(
    stage1_path: Path,
    *,
    candidate_ids: list[str],
    vectors: np.ndarray,
    labels: np.ndarray,
    reduced: np.ndarray,
    manifest: ClusterManifest,
    n_clusters: int,
    noise_count: int,
    noise_ratio: float,
) -> Stage1ClusterArtifacts:
    stage1_path.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(stage1_path)

    np.save(paths["vectors"], vectors.astype(np.float32))
    np.save(paths["labels"], labels.astype(np.int32))
    np.save(paths["reduced"], reduced.astype(np.float32))
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return Stage1ClusterArtifacts(
        stage1_dir=stage1_path,
        candidate_ids=candidate_ids,
        vectors=vectors,
        labels=labels,
        reduced=reduced,
        manifest=manifest,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
    )


def load_cluster_artifacts(
    stage1_path: Path,
    candidate_ids: list[str],
) -> Stage1ClusterArtifacts:
    paths = _artifact_paths(stage1_path)
    missing = _missing_artifact_names(stage1_path)
    if missing:
        raise Stage1ArtifactsMissingError(
            f"Missing Stage 1 cluster artifacts in {stage1_path}: {', '.join(missing)}. "
            "Run precompute_stage1_clustering(artifacts_dir) first."
        )

    with open(paths["manifest"], encoding="utf-8") as f:
        manifest = ClusterManifest.from_dict(json.load(f))

    vectors = np.load(paths["vectors"]).astype(np.float32)
    labels = np.load(paths["labels"])
    reduced = np.load(paths["reduced"]).astype(np.float32)

    n = len(candidate_ids)
    if vectors.shape != (n, manifest.vector_dim):
        raise ValueError(
            f"candidate_vectors shape {vectors.shape} does not match "
            f"({n}, {manifest.vector_dim}) from id_map"
        )
    if labels.shape != (n,):
        raise ValueError(f"cluster_labels shape {labels.shape} does not match ({n},)")
    if reduced.shape != (n, manifest.clustering_dims):
        raise ValueError(
            f"umap_reduced shape {reduced.shape} does not match "
            f"({n}, {manifest.clustering_dims})"
        )

    unique_labels = set(int(x) for x in labels)
    n_clusters = len([label for label in unique_labels if label >= 0])
    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / len(labels) if len(labels) else 0.0

    return Stage1ClusterArtifacts(
        stage1_dir=stage1_path,
        candidate_ids=candidate_ids,
        vectors=vectors,
        labels=labels,
        reduced=reduced,
        manifest=manifest,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
    )


def require_cluster_artifacts(
    artifacts_dir: Path,
    candidate_ids: list[str],
    *,
    stage1_path: Path | None = None,
) -> Stage1ClusterArtifacts:
    resolved = stage1_path if stage1_path is not None else stage1_dir(artifacts_dir)
    return load_cluster_artifacts(resolved, candidate_ids)


def validate_manifest_params(
    manifest: ClusterManifest,
    *,
    random_seed: int,
    clustering_dims: int,
    n_neighbors: int,
) -> None:
    mismatches: list[str] = []
    if manifest.random_seed != random_seed:
        mismatches.append(
            f"random_seed (manifest={manifest.random_seed}, requested={random_seed})"
        )
    if manifest.clustering_dims != clustering_dims:
        mismatches.append(
            f"clustering_dims (manifest={manifest.clustering_dims}, "
            f"requested={clustering_dims})"
        )
    if manifest.n_neighbors != n_neighbors:
        mismatches.append(
            f"n_neighbors (manifest={manifest.n_neighbors}, requested={n_neighbors})"
        )
    if mismatches:
        raise ValueError(
            "Stage 1 cluster manifest does not match filter parameters: "
            + "; ".join(mismatches)
            + ". Re-run precompute_stage1_clustering() with matching params."
        )
