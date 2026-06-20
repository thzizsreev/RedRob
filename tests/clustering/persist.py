"""Persist clustering pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.clustering.cluster import ClusteringResult
from tests.clustering.reduce import ReductionResult


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def persist_artifacts(
    output_dir: Path,
    *,
    sample_ids: list[str],
    reduction: ReductionResult,
    labels: np.ndarray,
    label_map: dict[str, int],
    metrics: ClusteringResult,
    cluster_inspection: dict,
    noise_export: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "sample_ids.json", sample_ids)
    save_numpy(output_dir / "umap_clustering_12d.npy", reduction.umap_clustering)
    save_numpy(output_dir / "umap_viz_2d.npy", reduction.umap_viz)
    save_numpy(output_dir / "pca_viz_2d.npy", reduction.pca_viz)
    save_json(
        output_dir / "pca_summary.json",
        {
            "explained_variance_ratio_2d": reduction.pca_explained_variance_ratio,
            "cumulative_variance_first_3_components": reduction.pca_cumulative_variance_3,
        },
    )
    save_json(output_dir / "cluster_labels.json", label_map)
    save_json(
        output_dir / "metrics_summary.json",
        {
            "silhouette_score": metrics.silhouette,
            "noise_count": metrics.noise_count,
            "noise_ratio": metrics.noise_ratio,
            "n_clusters": metrics.n_clusters,
            "min_cluster_size": metrics.min_cluster_size,
            "min_samples": metrics.min_samples,
            "cluster_sizes": {str(k): v for k, v in metrics.cluster_sizes.items()},
        },
    )
    save_json(output_dir / "cluster_inspection.json", cluster_inspection)
    save_json(output_dir / "noise_candidates.json", noise_export)
