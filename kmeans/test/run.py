#!/usr/bin/env python3
"""
K-means Phase B — cluster, visualize, and persist (fast re-runs).

Requires Phase A artifacts from kmeans/precompute.py. Edit config below.
Outputs under kmeans/precomputed/<pool>/runs/k<N>/:
  cluster_plot.html, cluster_labels.json, metrics_summary.json, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kmeans.io import load_records_for_ids
from kmeans.precompute_artifacts import require_precompute_artifacts
from kmeans.test.assign import cluster_candidates_kmeans
from kmeans.test.cluster_inspect import build_cluster_inspection
from kmeans.test.metrics import compute_metrics
from kmeans.test.persist import persist_run_artifacts
from kmeans.test.visualize import write_cluster_plot, write_landmark_plot
from tracks.shared.paths import ROOT_DIR

# --- edit before run ---
POOL_TAG = "candidates_full"
PRECOMPUTE_DIR = ROOT_DIR / "kmeans" / "precomputed" / POOL_TAG

N_CLUSTERS = 1000
RANDOM_SEED = 42
LANDMARK_CANDIDATE_IDS: list[str] = []
ENABLE_ID_SEARCH = True

OUTPUT_DIR = PRECOMPUTE_DIR / "runs" / f"k{N_CLUSTERS}"


def run_kmeans_cluster(
    precompute_dir: Path,
    output_dir: Path,
    *,
    n_clusters: int,
    random_seed: int,
    landmark_ids: list[str],
    enable_id_search: bool,
) -> None:
    print("Phase B — K-means cluster + visualize")
    print(f"Precompute: {precompute_dir}")
    print(f"Output:     {output_dir}")

    precompute = require_precompute_artifacts(precompute_dir)
    manifest = precompute.manifest

    if manifest.random_seed != random_seed:
        print(
            f"WARNING: RANDOM_SEED={random_seed} differs from precompute "
            f"manifest ({manifest.random_seed})"
        )

    candidate_ids = precompute.candidate_ids
    records = load_records_for_ids(Path(manifest.candidates_path), candidate_ids)
    print(f"Loaded {len(candidate_ids):,} candidates from precompute")

    print(f"Running K-Means (n_clusters={n_clusters})...")
    labels, inertia = cluster_candidates_kmeans(
        precompute.umap_clustering,
        n_clusters=n_clusters,
        random_state=random_seed,
    )
    metrics = compute_metrics(precompute.umap_clustering, labels, inertia)

    label_map = {
        candidate_id: int(label)
        for candidate_id, label in zip(candidate_ids, labels)
    }
    cluster_inspection = build_cluster_inspection(
        candidate_ids,
        records,
        labels,
        random_seed=random_seed,
    )

    print("Writing plots and artifacts...")
    write_cluster_plot(
        precompute.umap_viz,
        candidate_ids,
        records,
        labels,
        output_dir / "cluster_plot.html",
        enable_id_search=enable_id_search,
    )
    if landmark_ids:
        write_landmark_plot(
            precompute.umap_viz,
            candidate_ids,
            records,
            labels,
            landmark_ids,
            output_dir / "landmark_plot.html",
            enable_id_search=enable_id_search,
        )

    run_manifest = {
        "precompute_dir": str(precompute_dir),
        "output_dir": str(output_dir),
        "n_clusters": n_clusters,
        "random_seed": random_seed,
        "pool_size": len(candidate_ids),
        "landmark_candidate_ids": landmark_ids,
        "precompute_manifest": manifest.to_dict(),
    }
    persist_run_artifacts(
        output_dir,
        candidate_ids=candidate_ids,
        labels=labels,
        label_map=label_map,
        metrics=metrics,
        cluster_inspection=cluster_inspection,
        manifest=run_manifest,
    )

    print("\n--- Metrics ---")
    print(f"Clusters:   {metrics.n_clusters}")
    print(f"Inertia:    {metrics.inertia:.2f}")
    if metrics.silhouette is not None:
        print(f"Silhouette: {metrics.silhouette:.4f}")
    else:
        print("Silhouette: n/a (need >=2 clusters)")
    print(f"PCA var (3 comp): {precompute.pca_cumulative_variance_3:.1%}")
    print(f"\nDone. Open {output_dir / 'cluster_plot.html'}")


def main() -> None:
    start_time = perf_counter()
    run_kmeans_cluster(
        PRECOMPUTE_DIR,
        OUTPUT_DIR,
        n_clusters=N_CLUSTERS,
        random_seed=RANDOM_SEED,
        landmark_ids=LANDMARK_CANDIDATE_IDS,
        enable_id_search=ENABLE_ID_SEARCH,
    )
    elapsed = perf_counter() - start_time
    print(f"K-means run completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
