"""Stage 2 I/O — load Stage 1 filter artifacts, candidates subset, write outputs."""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

from tracks.instructor.core.config import STAGE1_CLUSTER_LABELS_FILENAME, STAGE1_UMAP_REDUCED_FILENAME
from tracks.instructor.core.io import iter_candidates_from_path, load_candidate_ids_from_id_map

KMEANS_CLUSTER_LABELS_FILENAME = "cluster_labels.npy"
KMEANS_UMAP_CLUSTERING_FILENAME = "umap_clustering_15d.npy"


def load_stage1_filter(
    stage1_path: Path,
) -> tuple[list[str], dict[str, dict]]:
    ids_path = stage1_path / "filtered_ids.json"
    meta_path = stage1_path / "filtered_metadata.json"

    if not ids_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 filter output: {ids_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 metadata: {meta_path}")

    with open(ids_path, encoding="utf-8") as f:
        filtered_ids = json.load(f)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    return filtered_ids, metadata


def _resolve_cluster_artifact_paths(stage1_path: Path) -> tuple[Path | None, Path | None]:
    """
    Locate cluster_labels + UMAP reduced arrays for dist_to_centroid.

    Supports:
      - Instructor HDBSCAN:  <pool>/stage1/{cluster_labels,umap_reduced_12d}.npy
      - K-means filter:      .../runs/k<N>/filter/  + labels in parent run dir,
                             UMAP in precompute pool dir
    """
    hdbscan_labels = stage1_path / STAGE1_CLUSTER_LABELS_FILENAME
    hdbscan_reduced = stage1_path / STAGE1_UMAP_REDUCED_FILENAME
    if hdbscan_labels.exists() and hdbscan_reduced.exists():
        return hdbscan_labels, hdbscan_reduced

    if stage1_path.name == "filter":
        run_dir = stage1_path.parent
        precompute_dir = run_dir.parent.parent
        kmeans_labels = run_dir / KMEANS_CLUSTER_LABELS_FILENAME
        kmeans_reduced = precompute_dir / KMEANS_UMAP_CLUSTERING_FILENAME
        if kmeans_labels.exists() and kmeans_reduced.exists():
            return kmeans_labels, kmeans_reduced

    return None, None


def compute_dist_to_centroid(
    filtered_ids: list[str],
    artifacts_path: Path,
    stage1_path: Path,
) -> dict[str, float | None]:
    """L2 distance to cluster centroid in reduced clustering space (filtered IDs only)."""
    labels_path, reduced_path = _resolve_cluster_artifact_paths(stage1_path)

    if labels_path is None or reduced_path is None:
        warnings.warn(
            f"No cluster_labels/UMAP artifacts found for {stage1_path}. "
            "dist_to_centroid will be null for all survivors. "
            "For HDBSCAN: run tracks/instructor/stage0/run_cluster.py. "
            "For K-means: run kmeans/test/run.py then kmeans/filter/run.py.",
            stacklevel=2,
        )
        return {cid: None for cid in filtered_ids}

    candidate_ids = load_candidate_ids_from_id_map(artifacts_path)
    id_to_row = {cid: i for i, cid in enumerate(candidate_ids)}

    labels = np.load(labels_path)
    reduced = np.load(reduced_path).astype(np.float32)

    centroids: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        label_int = int(label)
        if label_int < 0:
            continue
        mask = labels == label
        centroids[label_int] = reduced[mask].mean(axis=0)

    result: dict[str, float | None] = {}
    for cid in filtered_ids:
        row = id_to_row.get(cid)
        if row is None:
            result[cid] = None
            continue
        label_int = int(labels[row])
        if label_int < 0:
            result[cid] = None
            continue
        centroid = centroids.get(label_int)
        if centroid is None:
            result[cid] = None
            continue
        dist = float(np.linalg.norm(reduced[row] - centroid))
        result[cid] = dist

    return result


def iter_candidates_by_ids(
    path: Path,
    id_set: set[str],
) -> Iterator[dict]:
    """Stream candidates.jsonl; yield only records in id_set; stop when all found."""
    if not id_set:
        return

    remaining = set(id_set)
    for record in iter_candidates_from_path(path):
        cid = record.get("candidate_id")
        if cid not in remaining:
            continue
        yield record
        remaining.discard(str(cid))
        if not remaining:
            break

    if remaining:
        missing = sorted(remaining)[:5]
        raise ValueError(
            f"Missing {len(remaining)} candidate(s) in {path}. "
            f"Examples: {missing}"
        )


def _write_gated_json(path: Path, survivors_df: pl.DataFrame) -> None:
    """Pretty-printed JSON mirror of stage2_gated.parquet for inspection."""
    records = survivors_df.to_dicts() if survivors_df.height > 0 else []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def write_stage2_outputs(
    output_dir: Path,
    survivors_df: pl.DataFrame,
    honeypot_log: list[dict],
    removed_log: list[dict],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    survivors_df.write_parquet(output_dir / "stage2_gated.parquet")
    _write_gated_json(output_dir / "stage2_gated.json", survivors_df)

    filtered_ids = (
        survivors_df["candidate_id"].to_list() if survivors_df.height > 0 else []
    )
    with open(output_dir / "stage2_filtered_ids.json", "w", encoding="utf-8") as f:
        json.dump(filtered_ids, f, indent=2)
        f.write("\n")

    with open(output_dir / "stage2_honeypot_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rules", "details_json"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in honeypot_log:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rules": "|".join(row.get("rules", [])),
                    "details_json": json.dumps(row.get("details", {})),
                }
            )

    with open(output_dir / "stage2_removed_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "reason_code"])
        writer.writeheader()
        writer.writerows(removed_log)

    with open(output_dir / "stage2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
