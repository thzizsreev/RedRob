"""Orchestrate semantic richness clustering stages."""

from __future__ import annotations

from pathlib import Path

from tests.clustering.cluster import compute_metrics, run_hdbscan
from tests.clustering.inspect import build_cluster_inspection
from tests.clustering.io import load_vectors_and_records
from tests.clustering.outliers import build_noise_export
from tests.clustering.persist import persist_artifacts
from tests.clustering.reduce import reduce_embeddings
from tests.clustering.sampling import stratified_sample
from tests.clustering.visualize import write_cluster_plot, write_landmark_plot
from tracks.instructor.core.config import INDEX_FILENAME, VECTOR_DIM


def run_clustering_pipeline(
    candidates_path: Path,
    artifacts_path: Path,
    output_dir: Path,
    *,
    sample_size: int,
    random_seed: int,
    landmark_ids: list[str],
    enable_id_search: bool = True,
    index_filename: str | None = None,
    vector_dim: int | None = None,
) -> None:
    print(f"Candidates: {candidates_path}")
    print(f"Vectors:    {artifacts_path}")
    print(f"Output:     {output_dir}")

    inputs = load_vectors_and_records(
        candidates_path,
        artifacts_path,
        index_filename=index_filename or INDEX_FILENAME,
        vector_dim=vector_dim if vector_dim is not None else VECTOR_DIM,
    )
    print(f"Loaded {len(inputs.candidate_ids):,} indexed candidates")

    sample = stratified_sample(
        inputs,
        sample_size=sample_size,
        random_seed=random_seed,
        landmark_ids=landmark_ids,
    )
    print(f"Sampled {len(sample.candidate_ids):,} candidates (target={sample_size})")

    print("Running UMAP + PCA...")
    reduction = reduce_embeddings(sample.vectors, random_seed=random_seed)

    print("Running HDBSCAN...")
    labels, min_cluster_size, min_samples = run_hdbscan(
        reduction.umap_clustering,
        sample_size=len(sample.candidate_ids),
    )
    metrics = compute_metrics(
        reduction.umap_clustering,
        labels,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    label_map = {
        candidate_id: int(label)
        for candidate_id, label in zip(sample.candidate_ids, labels)
    }
    cluster_inspection = build_cluster_inspection(
        sample.candidate_ids,
        sample.records,
        labels,
        random_seed=random_seed,
    )
    noise_export = build_noise_export(
        sample.candidate_ids,
        sample.records,
        labels,
    )

    print("Writing plots and artifacts...")
    write_cluster_plot(
        reduction.umap_viz,
        sample.candidate_ids,
        sample.records,
        labels,
        output_dir / "cluster_plot.html",
        enable_id_search=enable_id_search,
    )
    write_landmark_plot(
        reduction.umap_viz,
        sample.candidate_ids,
        sample.records,
        labels,
        landmark_ids,
        output_dir / "landmark_plot.html",
        enable_id_search=enable_id_search,
    )
    persist_artifacts(
        output_dir,
        sample_ids=sample.candidate_ids,
        reduction=reduction,
        labels=labels,
        label_map=label_map,
        metrics=metrics,
        cluster_inspection=cluster_inspection,
        noise_export=noise_export,
    )

    print("\n--- Metrics ---")
    print(f"Clusters:   {metrics.n_clusters}")
    print(f"Noise:      {metrics.noise_count} ({metrics.noise_ratio:.1%})")
    if metrics.silhouette is not None:
        print(f"Silhouette: {metrics.silhouette:.4f}")
    else:
        print("Silhouette: n/a (need >=2 clusters with non-noise points)")
    print(f"PCA var (3 comp): {reduction.pca_cumulative_variance_3:.1%}")
    print(f"\nDone. Open {output_dir / 'cluster_plot.html'}")
