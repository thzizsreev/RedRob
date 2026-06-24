"""BM25 sparse retrieval for Q4 jargon query."""

from __future__ import annotations

import numpy as np
import polars as pl
from rank_bm25 import BM25Okapi


def bm25_retrieve_q4(
    bm25: BM25Okapi,
    q4_tokens: list[str],
    survivor_row_indices: np.ndarray,
    row_to_id: list[str],
    k_sparse: int,
) -> pl.DataFrame:
    """Score survivors with BM25 and return top-k_sparse ranked list."""
    query_tokens = [t.lower() for t in q4_tokens]
    all_scores = bm25.get_scores(query_tokens)

    survivor_indices = survivor_row_indices.astype(np.int64)
    survivor_scores = all_scores[survivor_indices]

    k = min(k_sparse, len(survivor_indices))
    if k == 0:
        return pl.DataFrame(
            schema={
                "candidate_id": pl.Utf8,
                "bm25_score": pl.Float64,
                "bm25_rank": pl.Int64,
            }
        )

    top_local = np.argsort(-survivor_scores)[:k]
    rows: list[dict] = []
    for rank, local_idx in enumerate(top_local, start=1):
        row_idx = int(survivor_indices[local_idx])
        rows.append(
            {
                "candidate_id": row_to_id[row_idx],
                "bm25_score": float(survivor_scores[local_idx]),
                "bm25_rank": rank,
            }
        )

    return pl.DataFrame(rows)
