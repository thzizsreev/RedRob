"""UMAP (clustering + visualization) and PCA diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import umap
from sklearn.decomposition import PCA


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
    clustering_dims: int = 15,
    n_neighbors: int = 20,
) -> ReductionResult:
    umap_clustering = umap.UMAP(
        n_components=clustering_dims,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=random_seed,
    ).fit_transform(vectors).astype(np.float32)

    umap_viz = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.2,
        metric="cosine",
        random_state=random_seed,
    ).fit_transform(vectors).astype(np.float32)

    pca = PCA(n_components=2, random_state=random_seed)
    pca_viz = pca.fit_transform(vectors).astype(np.float32)
    full_pca = PCA(n_components=min(3, vectors.shape[1]), random_state=random_seed)
    full_pca.fit(vectors)

    return ReductionResult(
        umap_clustering=umap_clustering,
        umap_viz=umap_viz,
        pca_viz=pca_viz,
        pca_explained_variance_ratio=[
            float(x) for x in pca.explained_variance_ratio_
        ],
        pca_cumulative_variance_3=float(full_pca.explained_variance_ratio_.sum()),
    )
