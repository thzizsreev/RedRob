"""Load K-means run labels and write filter JSON artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.kmeans.filter.pipeline import KMeansFilterResult

CLUSTER_LABELS_FILENAME = "cluster_labels.npy"
SAMPLE_IDS_FILENAME = "sample_ids.json"


class KMeansRunArtifactsMissingError(FileNotFoundError):
    """Raised when filter is run before a K-means cluster run exists."""


def load_run_labels(run_dir: Path, candidate_ids: list[str]) -> np.ndarray:
    labels_path = run_dir / CLUSTER_LABELS_FILENAME
    if not labels_path.exists():
        raise KMeansRunArtifactsMissingError(
            f"Missing {labels_path}. Run kmeans/test/run.py for this k first."
        )

    labels = np.load(labels_path)
    if labels.shape != (len(candidate_ids),):
        raise ValueError(
            f"cluster_labels shape {labels.shape} does not match "
            f"({len(candidate_ids)},) from precompute"
        )

    sample_ids_path = run_dir / SAMPLE_IDS_FILENAME
    if sample_ids_path.exists():
        with open(sample_ids_path, encoding="utf-8") as f:
            run_ids = json.load(f)
        if run_ids != candidate_ids:
            raise ValueError(
                f"sample_ids in {run_dir} do not match precompute candidate_ids"
            )

    return labels


def build_filtered_output(
    result: KMeansFilterResult,
    candidate_ids: list[str],
) -> tuple[list[str], dict[str, dict]]:
    filtered = result.filtered_ids
    id_to_index = {cid: i for i, cid in enumerate(candidate_ids)}

    label_to_ids: dict[int, list[str]] = defaultdict(list)
    for cid, label in zip(candidate_ids, result.labels):
        label_to_ids[int(label)].append(cid)

    cluster_rank_by_label = {
        label: rank for rank, (label, _median_sim, _size) in enumerate(result.ranked_clusters)
    }

    ordered_ids: list[str] = []
    metadata: dict[str, dict] = {}

    for label, median_sim, _size in result.ranked_clusters:
        cluster_ids = sorted(cid for cid in label_to_ids[label] if cid in filtered)
        if not cluster_ids:
            continue
        rank = cluster_rank_by_label[label]
        for cid in cluster_ids:
            ordered_ids.append(cid)
            idx = id_to_index[cid]
            metadata[cid] = {
                "cluster_id": label,
                "cluster_rank": rank,
                "cluster_median_similarity": median_sim,
                "anchor_similarity": float(result.anchor_similarities[idx]),
            }

    return ordered_ids, metadata


def write_filter_artifacts(
    output_dir: Path,
    result: KMeansFilterResult,
    *,
    candidate_ids: list[str],
    floor: int,
    n_clusters_k: int,
    precompute_dir: Path,
    run_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_ids, metadata = build_filtered_output(result, candidate_ids)

    with open(output_dir / "filtered_ids.json", "w", encoding="utf-8") as f:
        json.dump(ordered_ids, f, indent=2)

    with open(output_dir / "filtered_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    rankings = [
        {"label": label, "median_similarity": median_sim, "size": size}
        for label, median_sim, size in result.ranked_clusters
    ]
    with open(output_dir / "cluster_rankings.json", "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)

    summary = {
        "n_candidates": len(candidate_ids),
        "n_clusters": result.n_clusters,
        "n_clusters_k": n_clusters_k,
        "filtered_count": len(result.filtered_ids),
        "floor": floor,
        "precompute_dir": str(precompute_dir),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "filter_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_filter_summary(result: KMeansFilterResult, *, floor: int) -> None:
    print(f"\n--- Filter summary ---")
    print(f"Clusters:     {result.n_clusters}")
    print(f"Filtered set: {len(result.filtered_ids)} (floor={floor})")

    print("\n--- Ranked clusters (label, median_sim, size) ---")
    for label, median_sim, size in result.ranked_clusters[:10]:
        print(f"  {label:4d}  median={median_sim:.4f}  size={size}")
    if len(result.ranked_clusters) > 10:
        print(f"  ... ({len(result.ranked_clusters) - 10} more)")

    if len(result.filtered_ids) < floor:
        print(
            f"\nWARNING: filtered set ({len(result.filtered_ids)}) "
            f"is below floor ({floor})"
        )
