"""Stage 2 — UMAP (clustering + visualization) and PCA diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
import umap

from tracks.instructor.stage1.clustering import reduce_for_clustering
from tracks.instructor.core.config import UMAP_CLUSTERING_DIMS, UMAP_N_NEIGHBORS


@dataclass(frozen=True)
class ReductionResult:
    umap_clustering: np.ndarray
    umap_viz: np.ndarray
    pca_viz: np.ndarray
    pca_explained_variance_ratio: list[float]
    pca_cumulative_variance_3: float


def reduce_embeddings(
    vectors: np.ndarray,
    random_seed: int,
    *,
    clustering_dims: int = UMAP_CLUSTERING_DIMS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
) -> ReductionResult:
    umap_clustering = reduce_for_clustering(
        vectors,
        n_components=clustering_dims,
        random_state=random_seed,
        n_neighbors=n_neighbors,
    )

    umap_viz_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.2,
        metric="cosine",
        random_state=random_seed,
    )
    umap_viz = umap_viz_model.fit_transform(vectors)

    pca = PCA(n_components=2, random_state=random_seed)
    pca_viz = pca.fit_transform(vectors)
    full_pca = PCA(n_components=min(3, vectors.shape[1]), random_state=random_seed)
    full_pca.fit(vectors)

    return ReductionResult(
        umap_clustering=umap_clustering,
        umap_viz=umap_viz.astype(np.float32),
        pca_viz=pca_viz.astype(np.float32),
        pca_explained_variance_ratio=[
            float(x) for x in pca.explained_variance_ratio_
        ],
        pca_cumulative_variance_3=float(full_pca.explained_variance_ratio_.sum()),
    )
