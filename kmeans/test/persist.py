"""Persist K-means Phase B run artifacts (k-dependent outputs only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kmeans.test.metrics import ClusteringResult


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def persist_run_artifacts(
    output_dir: Path,
    *,
    candidate_ids: list[str],
    labels: np.ndarray,
    label_map: dict[str, int],
    metrics: ClusteringResult,
    cluster_inspection: dict,
    manifest: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "cluster_labels.json", label_map)
    save_numpy(output_dir / "cluster_labels.npy", labels)
    save_json(
        output_dir / "metrics_summary.json",
        {
            "n_clusters": metrics.n_clusters,
            "inertia": metrics.inertia,
            "silhouette_score": metrics.silhouette,
            "cluster_sizes": {str(k): v for k, v in metrics.cluster_sizes.items()},
        },
    )
    save_json(output_dir / "cluster_inspection.json", cluster_inspection)
    save_json(output_dir / "kmeans_run_manifest.json", manifest)
    save_json(output_dir / "sample_ids.json", candidate_ids)
