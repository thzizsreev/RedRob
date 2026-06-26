"""FAISS dense retrieval with IDSelector restricted to Stage 2 survivors."""

from __future__ import annotations

import faiss
import numpy as np
import polars as pl


def build_id_selector(survivor_row_indices: np.ndarray) -> faiss.IDSelectorBatch:
    sorted_ids = np.sort(survivor_row_indices.astype(np.int64))
    return faiss.IDSelectorBatch(sorted_ids)


def search_dense(
    index: faiss.Index,
    query_vec: np.ndarray,
    k: int,
    selector: faiss.IDSelectorBatch,
    row_to_id: list[str],
) -> pl.DataFrame:
    """Search FAISS index; return DataFrame with candidate_id, score, rank columns."""
    params = faiss.SearchParameters()
    params.sel = selector

    query_matrix = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(query_matrix, k, params=params)

    rows: list[dict] = []
    rank = 1
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        candidate_id = row_to_id[int(idx)]
        rows.append(
            {
                "candidate_id": candidate_id,
                "score": float(score),
                "rank": rank,
            }
        )
        rank += 1

    if not rows:
        return pl.DataFrame(schema={"candidate_id": pl.Utf8, "score": pl.Float64, "rank": pl.Int64})

    return pl.DataFrame(rows)


def dense_retrieve_q1(
    index: faiss.Index,
    query_vec: np.ndarray,
    k: int,
    selector: faiss.IDSelectorBatch,
    row_to_id: list[str],
) -> pl.DataFrame:
    hits = search_dense(index, query_vec, k, selector, row_to_id)
    return hits.rename({"score": "q1_score", "rank": "q1_rank"})


def dense_retrieve_q2(
    index: faiss.Index,
    query_vec: np.ndarray,
    k: int,
    selector: faiss.IDSelectorBatch,
    row_to_id: list[str],
) -> pl.DataFrame:
    hits = search_dense(index, query_vec, k, selector, row_to_id)
    return hits.rename({"score": "q2_score", "rank": "q2_rank"})
