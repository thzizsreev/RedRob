"""UMAP dimensionality reduction for clustering (production Stage 1)."""

from __future__ import annotations

import numpy as np
import umap


def reduce_for_clustering(
    vectors: np.ndarray,
    *,
    n_components: int = 12,
    random_state: int,
    n_neighbors: int = 20,
) -> np.ndarray:
    """Reduce high-dimensional embeddings for HDBSCAN clustering."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(vectors).astype(np.float32)
