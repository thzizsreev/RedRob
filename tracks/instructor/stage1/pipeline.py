"""Stage 1 cluster-based filtering — production runner and artifact writers."""

from __future__ import annotations

import json
import runpy
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if __package__ is None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.core.config import (
    INDEX_FILENAME,
    STAGE1_FLOOR,
    STAGE1_HDBSCAN_CORE_DIST_N_JOBS,
    STAGE1_RANDOM_SEED,
    STAGE1_UMAP_N_JOBS,
    UMAP_CLUSTERING_DIMS,
    UMAP_N_NEIGHBORS,
    VECTOR_DIM,
)
from tracks.instructor.core.io import (
    load_candidate_ids_from_id_map,
    load_jd_query_vector,
    load_vectors_from_artifacts,
)
from tracks.instructor.stage1.artifacts import (
    ClusterManifest,
    Stage1ClusterArtifacts,
    assert_cluster_artifacts_absent,
    cluster_artifacts_exist,
    require_cluster_artifacts,
    save_cluster_artifacts,
    stage1_dir,
    validate_manifest_params,
)
from tracks.instructor.stage1.clustering import min_cluster_size
from tracks.instructor.stage1.filtering.pipeline import (
    Stage1Result,
    cluster_candidates,
    filter_from_labels,
)


@dataclass(frozen=True)
class Stage1RunResult:
    result: Stage1Result
    candidate_ids: list[str]
    vectors: np.ndarray


def print_stage1_summary(result: Stage1Result, *, floor: int = STAGE1_FLOOR) -> None:
    print(f"\n--- Stage 1 summary ---")
    print(f"Clusters:     {result.n_clusters}")
    print(f"Noise:        {result.noise_count} ({result.noise_ratio:.1%})")
    print(f"Filtered set: {len(result.filtered_ids)} (floor={floor})")

    print("\n--- Ranked clusters (label, median_sim, size) ---")
    for label, median_sim, size in result.ranked_clusters:
        print(f"  {label:4d}  median={median_sim:.4f}  size={size}")

    if len(result.filtered_ids) < floor:
        print(
            f"\nWARNING: filtered set ({len(result.filtered_ids)}) "
            f"is below floor ({floor})"
        )


def build_filtered_output(
    result: Stage1Result,
    candidate_ids: list[str],
    anchor_similarities: np.ndarray,
) -> tuple[list[str], dict[str, dict]]:
    """
    Order filtered candidates by cluster rank (best cluster first), then by ID
    within each cluster. Noise points (-1) are appended last when included.
    """
    if result.anchor_similarities is None:
        raise ValueError("Stage1Result.anchor_similarities is required for metadata output")

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
        if label == -1:
            continue
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
                "anchor_similarity": float(anchor_similarities[idx]),
            }

    noise_ids = sorted(cid for cid in label_to_ids.get(-1, []) if cid in filtered)
    if noise_ids:
        noise_rank = cluster_rank_by_label.get(-1)
        noise_median = next(
            (median_sim for lbl, median_sim, _ in result.ranked_clusters if lbl == -1),
            None,
        )
        for cid in noise_ids:
            ordered_ids.append(cid)
            idx = id_to_index[cid]
            entry: dict = {
                "cluster_id": -1,
                "anchor_similarity": float(anchor_similarities[idx]),
            }
            if noise_rank is not None:
                entry["cluster_rank"] = noise_rank
            if noise_median is not None:
                entry["cluster_median_similarity"] = noise_median
            metadata[cid] = entry

    return ordered_ids, metadata


def write_stage1_artifacts(
    output_dir: Path,
    result: Stage1Result,
    *,
    candidate_ids: list[str],
    n_candidates: int,
    random_seed: int,
    floor: int = STAGE1_FLOOR,
) -> None:
    if result.anchor_similarities is None:
        raise ValueError("Stage1Result.anchor_similarities is required to write artifacts")

    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_ids, metadata = build_filtered_output(
        result, candidate_ids, result.anchor_similarities
    )

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
        "n_candidates": n_candidates,
        "n_clusters": result.n_clusters,
        "noise_count": result.noise_count,
        "noise_ratio": result.noise_ratio,
        "filtered_count": len(result.filtered_ids),
        "floor": floor,
        "random_seed": random_seed,
    }
    with open(output_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def precompute_stage1_clustering(
    artifacts_dir: Path,
    *,
    stage1_path: Path | None = None,
    random_seed: int = STAGE1_RANDOM_SEED,
    clustering_dims: int = UMAP_CLUSTERING_DIMS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    umap_n_jobs: int = STAGE1_UMAP_N_JOBS,
    hdbscan_core_dist_n_jobs: int = STAGE1_HDBSCAN_CORE_DIST_N_JOBS,
    index_filename: str = INDEX_FILENAME,
    vector_dim: int = VECTOR_DIM,
    overwrite: bool = False,
    print_summary: bool = True,
) -> Stage1ClusterArtifacts:
    """
    Phase A: export vectors from FAISS, UMAP + HDBSCAN, write .npy artifacts.
    """
    resolved_stage1 = stage1_path if stage1_path is not None else stage1_dir(artifacts_dir)
    assert_cluster_artifacts_absent(resolved_stage1, overwrite=overwrite)

    if print_summary:
        print(f"Phase A — cluster precompute")
        print(f"Artifacts: {artifacts_dir}")
        print(f"Output:    {resolved_stage1}")

    candidate_ids, vectors = load_vectors_from_artifacts(
        artifacts_dir,
        index_filename=index_filename,
        vector_dim=vector_dim,
    )
    if print_summary:
        print(f"Loaded {len(candidate_ids):,} candidates from FAISS")

    reduced, labels = cluster_candidates(
        vectors,
        random_seed=random_seed,
        clustering_dims=clustering_dims,
        n_neighbors=n_neighbors,
        umap_n_jobs=umap_n_jobs,
        hdbscan_core_dist_n_jobs=hdbscan_core_dist_n_jobs,
    )

    sample_size = len(candidate_ids)
    mcs = min_cluster_size(sample_size)
    unique_labels = set(int(x) for x in labels)
    n_clusters = len([label for label in unique_labels if label >= 0])
    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / len(labels) if len(labels) else 0.0

    manifest = ClusterManifest(
        n_candidates=sample_size,
        vector_dim=vector_dim,
        random_seed=random_seed,
        clustering_dims=clustering_dims,
        n_neighbors=n_neighbors,
        umap_n_jobs=umap_n_jobs,
        hdbscan_core_dist_n_jobs=hdbscan_core_dist_n_jobs,
        min_cluster_size=mcs,
    )

    artifacts = save_cluster_artifacts(
        resolved_stage1,
        candidate_ids=candidate_ids,
        vectors=vectors,
        labels=labels,
        reduced=reduced,
        manifest=manifest,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
    )

    if print_summary:
        print(f"Clusters: {n_clusters}")
        print(f"Noise:    {noise_count} ({noise_ratio:.1%})")
        print(f"Wrote cluster artifacts to {resolved_stage1}")

    return artifacts


def run_stage1_filter(
    artifacts_dir: Path,
    *,
    stage1_path: Path | None = None,
    output_dir: Path | None = None,
    anchor_vec: np.ndarray | None = None,
    floor: int = STAGE1_FLOOR,
    random_seed: int = STAGE1_RANDOM_SEED,
    clustering_dims: int = UMAP_CLUSTERING_DIMS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    index_filename: str = INDEX_FILENAME,
    print_summary: bool = True,
    write_artifacts: bool = True,
) -> Stage1RunResult:
    """
    Phase B: load phase-A .npy artifacts, rank clusters, filter, write JSON.
    """
    resolved_stage1 = stage1_path if stage1_path is not None else stage1_dir(artifacts_dir)

    if print_summary:
        print(f"Phase B — filter")
        print(f"Artifacts: {artifacts_dir}")
        print(f"Stage1:    {resolved_stage1}")

    candidate_ids = load_candidate_ids_from_id_map(
        artifacts_dir, index_filename=index_filename
    )
    cluster_data = require_cluster_artifacts(
        artifacts_dir, candidate_ids, stage1_path=resolved_stage1
    )
    validate_manifest_params(
        cluster_data.manifest,
        random_seed=random_seed,
        clustering_dims=clustering_dims,
        n_neighbors=n_neighbors,
    )

    vectors = cluster_data.vectors
    labels = cluster_data.labels

    if anchor_vec is None:
        anchor_vec = load_jd_query_vector(artifacts_dir)

    result = filter_from_labels(
        candidate_ids,
        vectors,
        labels,
        anchor_vec,
        floor=floor,
    )

    if print_summary:
        print_stage1_summary(result, floor=floor)

    if write_artifacts:
        resolved_output = output_dir if output_dir is not None else resolved_stage1
        write_stage1_artifacts(
            resolved_output,
            result,
            candidate_ids=candidate_ids,
            n_candidates=len(candidate_ids),
            random_seed=random_seed,
            floor=floor,
        )
        if print_summary:
            print(f"\nWrote filter artifacts to {resolved_output}")

    return Stage1RunResult(
        result=result,
        candidate_ids=candidate_ids,
        vectors=vectors,
    )


def run_stage1_from_artifacts(
    artifacts_dir: Path,
    *,
    output_dir: Path | None = None,
    random_seed: int = STAGE1_RANDOM_SEED,
    floor: int = STAGE1_FLOOR,
    index_filename: str = INDEX_FILENAME,
    vector_dim: int = VECTOR_DIM,
    anchor_vec: np.ndarray | None = None,
    print_summary: bool = True,
    overwrite_cluster: bool = False,
) -> Stage1RunResult:
    """
    Deprecated convenience wrapper: runs phase A (if needed) then phase B.

    Prefer explicit precompute_stage1_clustering() + run_stage1_filter().
    """
    warnings.warn(
        "run_stage1_from_artifacts() is deprecated. "
        "Use precompute_stage1_clustering() and run_stage1_filter() separately.",
        DeprecationWarning,
        stacklevel=2,
    )

    resolved_stage1 = output_dir if output_dir is not None else stage1_dir(artifacts_dir)

    if overwrite_cluster or not cluster_artifacts_exist(resolved_stage1):
        precompute_stage1_clustering(
            artifacts_dir,
            stage1_path=resolved_stage1,
            random_seed=random_seed,
            index_filename=index_filename,
            vector_dim=vector_dim,
            overwrite=overwrite_cluster,
            print_summary=print_summary,
        )
    elif print_summary:
        print(f"Using existing cluster artifacts in {resolved_stage1}")

    return run_stage1_filter(
        artifacts_dir,
        stage1_path=resolved_stage1,
        output_dir=resolved_stage1,
        anchor_vec=anchor_vec,
        floor=floor,
        random_seed=random_seed,
        index_filename=index_filename,
        print_summary=print_summary,
    )


if __name__ == "__main__":
    runpy.run_path(str(_ROOT / "tracks/instructor/stage1/run_filter.py"), run_name="__main__")
