"""Stage 1 cluster-based filtering — production runner and artifact writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracks.instructor.config import (
    INDEX_FILENAME,
    STAGE1_FLOOR,
    STAGE1_RANDOM_SEED,
    VECTOR_DIM,
)
from tracks.instructor.filtering.pipeline import Stage1Result, run_stage1_filtering
from tracks.instructor.io import load_jd_query_vector, load_vectors_from_artifacts


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


def write_stage1_artifacts(
    output_dir: Path,
    result: Stage1Result,
    *,
    n_candidates: int,
    random_seed: int,
    floor: int = STAGE1_FLOOR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "filtered_ids.json", "w", encoding="utf-8") as f:
        json.dump(sorted(result.filtered_ids), f, indent=2)

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
) -> Stage1RunResult:
    """
    Load precomputed vectors, run Stage 1 filtering, optionally write JSON artifacts.

    When anchor_vec is None, loads jd_query_vec.npy from artifacts_dir.
    """
    if output_dir is not None:
        print(f"Artifacts: {artifacts_dir}")
        print(f"Output:    {output_dir}")

    candidate_ids, vectors = load_vectors_from_artifacts(
        artifacts_dir,
        index_filename=index_filename,
        vector_dim=vector_dim,
    )
    if anchor_vec is None:
        anchor_vec = load_jd_query_vector(artifacts_dir)

    if print_summary and output_dir is not None:
        print(f"Loaded {len(candidate_ids):,} candidates")

    result = run_stage1_filtering(
        candidate_ids,
        vectors,
        anchor_vec,
        random_seed=random_seed,
        floor=floor,
    )

    if print_summary:
        if output_dir is None:
            print(
                f"Stage 1: {len(result.filtered_ids)} candidates "
                f"(floor={floor}, clusters={result.n_clusters}, "
                f"noise={result.noise_count}, {result.noise_ratio:.1%})"
            )
            if result.ranked_clusters:
                top_label, top_sim, top_size = result.ranked_clusters[0]
                print(
                    f"  Top cluster: label={top_label}, "
                    f"median_sim={top_sim:.4f}, size={top_size}"
                )
        else:
            print_stage1_summary(result, floor=floor)

    if output_dir is not None:
        write_stage1_artifacts(
            output_dir,
            result,
            n_candidates=len(candidate_ids),
            random_seed=random_seed,
            floor=floor,
        )
        print(f"\nWrote artifacts to {output_dir}")

    return Stage1RunResult(
        result=result,
        candidate_ids=candidate_ids,
        vectors=vectors,
    )
