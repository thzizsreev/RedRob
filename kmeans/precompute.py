#!/usr/bin/env python3
"""
K-means Phase A — UMAP/PCA precompute.

Run once per candidate pool after precompute.py (FAISS index). Edit config below.
Outputs under kmeans/precomputed/<pool>/:
  precompute_manifest.json, sample_ids.json, candidate_vectors.npy
  umap_clustering_15d.npy, umap_viz_2d.npy, pca_viz_2d.npy, pca_summary.json

Then run kmeans/test/run.py (Phase B) to cluster with different N_CLUSTERS values.
Then run kmeans/filter/run.py (Phase C) to rank clusters and filter to floor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kmeans.io import load_inputs
from kmeans.precompute_artifacts import (
    PrecomputeManifest,
    assert_precompute_artifacts_absent,
    save_precompute_artifacts,
)
from kmeans.test.reduce import reduce_embeddings
from kmeans.test.sampling import Sample, subsample_inputs
from tracks.instructor.config import INDEX_FILENAME, VECTOR_DIM
from tracks.shared.paths import ARTIFACTS_DIR, ROOT_DIR, SAMPLE1K_PATH, CANDIDATES_JSONL_PATH

# --- edit before run ---
POOL_TAG = "candidates_full"
CANDIDATES_PATH = CANDIDATES_JSONL_PATH
ARTIFACTS_PATH = ARTIFACTS_DIR / POOL_TAG
PRECOMPUTE_DIR = ROOT_DIR / "kmeans" / "precomputed" / POOL_TAG

RANDOM_SEED = 42
SAMPLE_SIZE: int | None = None
CLUSTERING_DIMS = 15
N_NEIGHBORS = 20
OVERWRITE = False


def precompute_kmeans_reduction(
    candidates_path: Path,
    artifacts_path: Path,
    precompute_dir: Path,
    *,
    random_seed: int,
    sample_size: int | None,
    clustering_dims: int = CLUSTERING_DIMS,
    n_neighbors: int = N_NEIGHBORS,
    overwrite: bool = False,
    print_summary: bool = True,
) -> None:
    assert_precompute_artifacts_absent(precompute_dir, overwrite=overwrite)

    if print_summary:
        print("Phase A — K-means precompute (UMAP + PCA)")
        print(f"Candidates: {candidates_path}")
        print(f"Vectors:    {artifacts_path}")
        print(f"Output:     {precompute_dir}")

    inputs = load_inputs(
        candidates_path,
        artifacts_path,
        index_filename=INDEX_FILENAME,
        vector_dim=VECTOR_DIM,
    )
    if print_summary:
        print(f"Loaded {len(inputs.candidate_ids):,} indexed candidates")

    if sample_size is not None:
        sample = subsample_inputs(inputs, sample_size, random_seed)
        if print_summary:
            print(
                f"Sampled {len(sample.candidate_ids):,} candidates "
                f"(target={sample_size})"
            )
    else:
        sample = Sample(
            candidate_ids=inputs.candidate_ids,
            records=inputs.records,
            vectors=inputs.vectors,
        )

    if print_summary:
        print("Running UMAP + PCA...")
    reduction = reduce_embeddings(
        sample.vectors,
        random_seed=random_seed,
        clustering_dims=clustering_dims,
        n_neighbors=n_neighbors,
    )

    manifest = PrecomputeManifest(
        n_candidates=len(sample.candidate_ids),
        vector_dim=sample.vectors.shape[1],
        random_seed=random_seed,
        clustering_dims=clustering_dims,
        n_neighbors=n_neighbors,
        candidates_path=str(candidates_path),
        artifacts_path=str(artifacts_path),
        sample_size=sample_size,
    )

    save_precompute_artifacts(
        precompute_dir,
        candidate_ids=sample.candidate_ids,
        vectors=sample.vectors,
        umap_clustering=reduction.umap_clustering,
        umap_viz=reduction.umap_viz,
        pca_viz=reduction.pca_viz,
        pca_explained_variance_ratio=reduction.pca_explained_variance_ratio,
        pca_cumulative_variance_3=reduction.pca_cumulative_variance_3,
        manifest=manifest,
    )

    if print_summary:
        print(f"PCA var (3 comp): {reduction.pca_cumulative_variance_3:.1%}")
        print(f"Wrote precompute artifacts to {precompute_dir}")


def main() -> None:
    start_time = perf_counter()
    precompute_kmeans_reduction(
        CANDIDATES_PATH,
        ARTIFACTS_PATH,
        PRECOMPUTE_DIR,
        random_seed=RANDOM_SEED,
        sample_size=SAMPLE_SIZE,
        clustering_dims=CLUSTERING_DIMS,
        n_neighbors=N_NEIGHBORS,
        overwrite=OVERWRITE,
    )
    elapsed = perf_counter() - start_time
    print(f"K-means precompute completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
