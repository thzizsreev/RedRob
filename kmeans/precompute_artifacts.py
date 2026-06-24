"""Save/load for K-means Phase A precompute artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRECOMPUTE_MANIFEST_FILENAME = "precompute_manifest.json"
SAMPLE_IDS_FILENAME = "sample_ids.json"
CANDIDATE_VECTORS_FILENAME = "candidate_vectors.npy"
UMAP_CLUSTERING_FILENAME = "umap_clustering_15d.npy"
UMAP_VIZ_FILENAME = "umap_viz_2d.npy"
PCA_VIZ_FILENAME = "pca_viz_2d.npy"
PCA_SUMMARY_FILENAME = "pca_summary.json"


class KMeansPrecomputeMissingError(FileNotFoundError):
    """Raised when Phase B is run before Phase A precompute."""


class KMeansPrecomputeExistsError(FileExistsError):
    """Raised when Phase A artifacts already exist and overwrite=False."""


@dataclass(frozen=True)
class PrecomputeManifest:
    n_candidates: int
    vector_dim: int
    random_seed: int
    clustering_dims: int
    n_neighbors: int
    candidates_path: str
    artifacts_path: str
    sample_size: int | None

    def to_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "vector_dim": self.vector_dim,
            "random_seed": self.random_seed,
            "clustering_dims": self.clustering_dims,
            "n_neighbors": self.n_neighbors,
            "candidates_path": self.candidates_path,
            "artifacts_path": self.artifacts_path,
            "sample_size": self.sample_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PrecomputeManifest:
        sample_size = data.get("sample_size")
        return cls(
            n_candidates=int(data["n_candidates"]),
            vector_dim=int(data.get("vector_dim", 2304)),
            random_seed=int(data["random_seed"]),
            clustering_dims=int(data["clustering_dims"]),
            n_neighbors=int(data["n_neighbors"]),
            candidates_path=str(data["candidates_path"]),
            artifacts_path=str(data["artifacts_path"]),
            sample_size=int(sample_size) if sample_size is not None else None,
        )


@dataclass(frozen=True)
class PrecomputeArtifacts:
    precompute_dir: Path
    candidate_ids: list[str]
    vectors: np.ndarray
    umap_clustering: np.ndarray
    umap_viz: np.ndarray
    pca_viz: np.ndarray
    pca_explained_variance_ratio: list[float]
    pca_cumulative_variance_3: float
    manifest: PrecomputeManifest


def _artifact_paths(precompute_dir: Path) -> dict[str, Path]:
    return {
        "manifest": precompute_dir / PRECOMPUTE_MANIFEST_FILENAME,
        "sample_ids": precompute_dir / SAMPLE_IDS_FILENAME,
        "vectors": precompute_dir / CANDIDATE_VECTORS_FILENAME,
        "umap_clustering": precompute_dir / UMAP_CLUSTERING_FILENAME,
        "umap_viz": precompute_dir / UMAP_VIZ_FILENAME,
        "pca_viz": precompute_dir / PCA_VIZ_FILENAME,
        "pca_summary": precompute_dir / PCA_SUMMARY_FILENAME,
    }


def precompute_artifacts_exist(precompute_dir: Path) -> bool:
    paths = _artifact_paths(precompute_dir)
    return all(path.exists() for path in paths.values())


def _missing_artifact_names(precompute_dir: Path) -> list[str]:
    return [
        name for name, path in _artifact_paths(precompute_dir).items() if not path.exists()
    ]


def assert_precompute_artifacts_absent(precompute_dir: Path, *, overwrite: bool) -> None:
    if precompute_artifacts_exist(precompute_dir) and not overwrite:
        raise KMeansPrecomputeExistsError(
            f"K-means precompute artifacts already exist in {precompute_dir}. "
            "Set OVERWRITE=True or delete the directory before re-running precompute."
        )


def save_precompute_artifacts(
    precompute_dir: Path,
    *,
    candidate_ids: list[str],
    vectors: np.ndarray,
    umap_clustering: np.ndarray,
    umap_viz: np.ndarray,
    pca_viz: np.ndarray,
    pca_explained_variance_ratio: list[float],
    pca_cumulative_variance_3: float,
    manifest: PrecomputeManifest,
) -> PrecomputeArtifacts:
    precompute_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(precompute_dir)

    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    with open(paths["sample_ids"], "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f, indent=2)
    np.save(paths["vectors"], vectors.astype(np.float32))
    np.save(paths["umap_clustering"], umap_clustering.astype(np.float32))
    np.save(paths["umap_viz"], umap_viz.astype(np.float32))
    np.save(paths["pca_viz"], pca_viz.astype(np.float32))
    with open(paths["pca_summary"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "explained_variance_ratio_2d": pca_explained_variance_ratio,
                "cumulative_variance_first_3_components": pca_cumulative_variance_3,
            },
            f,
            indent=2,
        )

    return PrecomputeArtifacts(
        precompute_dir=precompute_dir,
        candidate_ids=candidate_ids,
        vectors=vectors.astype(np.float32),
        umap_clustering=umap_clustering.astype(np.float32),
        umap_viz=umap_viz.astype(np.float32),
        pca_viz=pca_viz.astype(np.float32),
        pca_explained_variance_ratio=pca_explained_variance_ratio,
        pca_cumulative_variance_3=pca_cumulative_variance_3,
        manifest=manifest,
    )


def load_precompute_artifacts(precompute_dir: Path) -> PrecomputeArtifacts:
    paths = _artifact_paths(precompute_dir)
    missing = _missing_artifact_names(precompute_dir)
    if missing:
        raise KMeansPrecomputeMissingError(
            f"Missing K-means precompute artifacts in {precompute_dir}: "
            f"{', '.join(missing)}. Run kmeans/precompute.py first."
        )

    with open(paths["manifest"], encoding="utf-8") as f:
        manifest = PrecomputeManifest.from_dict(json.load(f))
    with open(paths["sample_ids"], encoding="utf-8") as f:
        candidate_ids = json.load(f)
    with open(paths["pca_summary"], encoding="utf-8") as f:
        pca_summary = json.load(f)

    vectors = np.load(paths["vectors"]).astype(np.float32)
    umap_clustering = np.load(paths["umap_clustering"]).astype(np.float32)
    umap_viz = np.load(paths["umap_viz"]).astype(np.float32)
    pca_viz = np.load(paths["pca_viz"]).astype(np.float32)

    n = len(candidate_ids)
    if vectors.shape != (n, manifest.vector_dim):
        raise ValueError(
            f"candidate_vectors shape {vectors.shape} does not match "
            f"({n}, {manifest.vector_dim})"
        )
    if umap_clustering.shape != (n, manifest.clustering_dims):
        raise ValueError(
            f"umap_clustering shape {umap_clustering.shape} does not match "
            f"({n}, {manifest.clustering_dims})"
        )
    if umap_viz.shape != (n, 2):
        raise ValueError(f"umap_viz shape {umap_viz.shape} does not match ({n}, 2)")
    if pca_viz.shape != (n, 2):
        raise ValueError(f"pca_viz shape {pca_viz.shape} does not match ({n}, 2)")

    return PrecomputeArtifacts(
        precompute_dir=precompute_dir,
        candidate_ids=candidate_ids,
        vectors=vectors,
        umap_clustering=umap_clustering,
        umap_viz=umap_viz,
        pca_viz=pca_viz,
        pca_explained_variance_ratio=list(
            pca_summary["explained_variance_ratio_2d"]
        ),
        pca_cumulative_variance_3=float(
            pca_summary["cumulative_variance_first_3_components"]
        ),
        manifest=manifest,
    )


def require_precompute_artifacts(precompute_dir: Path) -> PrecomputeArtifacts:
    return load_precompute_artifacts(precompute_dir)
