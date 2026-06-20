#!/usr/bin/env python3
"""
Phase 2 — CPU-only retrieval (5-minute budget).

Loads precomputed flat vector index and JD query vector.
Optional Stage 1 cluster-based filtering before top-k retrieval.
No torch, no INSTRUCTOR, no GPU — FAISS / numpy IP search only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from tracks.instructor.config import QUERY_WEIGHTS
from tracks.instructor.io import load_index_and_id_map, load_jd_query_vector
from tracks.instructor.stage1 import run_stage1_from_artifacts
from tracks.shared.paths import ROOT_DIR

# --- edit before run ---
ARTIFACTS_PATH = ROOT_DIR / "artifacts" / "sample1k"
RESULTS_PATH = ROOT_DIR / "test_output" / "retrieval" / "retrieval_results_sample1k.json"


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    candidate_id: str
    score: float


def apply_query_weights(
    query_vector: np.ndarray,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
) -> np.ndarray:
    """Re-scale query blocks at search time without re-encoding."""
    w_r, w_i, w_e = weights
    blocks = np.split(query_vector, 3)
    base_weights = QUERY_WEIGHTS
    scaled = [
        blocks[0] * (w_r / base_weights[0]) if base_weights[0] else blocks[0],
        blocks[1] * (w_i / base_weights[1]) if base_weights[1] else blocks[1],
        blocks[2] * (w_e / base_weights[2]) if base_weights[2] else blocks[2],
    ]
    return np.concatenate(scaled).astype(np.float32)


def search_index(
    index: faiss.Index,
    query_vector: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_matrix = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    return index.search(query_matrix, k)


def hits_from_search(
    scores: np.ndarray,
    indices: np.ndarray,
    id_map: dict[int, str],
) -> list[RetrievalHit]:
    results: list[RetrievalHit] = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        candidate_id = id_map.get(int(idx), f"UNKNOWN_{idx}")
        results.append(
            RetrievalHit(rank=rank, candidate_id=candidate_id, score=float(score))
        )
    return results


def retrieve_on_subset(
    candidate_ids: list[str],
    vectors: np.ndarray,
    filtered_ids: set[str],
    query_vector: np.ndarray,
    k: int,
) -> list[RetrievalHit]:
    id_to_row = {cid: i for i, cid in enumerate(candidate_ids)}
    row_indices = [id_to_row[cid] for cid in filtered_ids if cid in id_to_row]
    if not row_indices:
        return []

    subset_vectors = vectors[row_indices]
    subset_ids = [candidate_ids[i] for i in row_indices]
    scores = subset_vectors @ query_vector
    k = min(k, len(scores))
    top_indices = np.argsort(-scores)[:k]

    return [
        RetrievalHit(
            rank=rank,
            candidate_id=subset_ids[int(idx)],
            score=float(scores[int(idx)]),
        )
        for rank, idx in enumerate(top_indices, start=1)
    ]


def retrieve(
    k: int = 300,
    *,
    artifacts_dir: Path = ARTIFACTS_PATH,
    weights: tuple[float, float, float] | None = None,
    use_stage1_filter: bool = True,
) -> list[RetrievalHit]:
    """
    Run block-weighted vector retrieval against the precomputed index.

    When use_stage1_filter is True, runs UMAP + HDBSCAN cluster filtering
    first and retrieves top-k only from the filtered candidate pool.
    """
    query_vector = load_jd_query_vector(artifacts_dir)
    if weights is not None and weights != QUERY_WEIGHTS:
        query_vector = apply_query_weights(query_vector, weights)

    if use_stage1_filter:
        stage1_run = run_stage1_from_artifacts(
            artifacts_dir,
            output_dir=None,
            anchor_vec=query_vector,
            print_summary=True,
        )
        return retrieve_on_subset(
            stage1_run.candidate_ids,
            stage1_run.vectors,
            stage1_run.result.filtered_ids,
            query_vector,
            k,
        )

    index, id_map = load_index_and_id_map(artifacts_dir)
    scores, indices = search_index(index, query_vector, k)
    return hits_from_search(scores, indices, id_map)


def retrieve_from_text(
    query_text: str,
    k: int = 10,
    **kwargs,
) -> list[RetrievalHit]:
    """Online text encoding is not supported — JD query is precomputed in precompute.py."""
    raise NotImplementedError(
        "retrieve_from_text requires runtime INSTRUCTOR encoding. "
        "Use retrieve() with the precomputed jd_query_vec.npy from precompute.py."
    )


def write_results_json(results: list[RetrievalHit], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"rank": hit.rank, "candidate_id": hit.candidate_id, "score": hit.score}
        for hit in results
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> list[RetrievalHit]:
    results = retrieve(k=300, artifacts_dir=ARTIFACTS_PATH)
    write_results_json(results, RESULTS_PATH)

    for hit in results[:10]:
        print(f"  {hit.rank:3d}. {hit.candidate_id}  score={hit.score:.4f}")
    print(f"Retrieval complete: {len(results)} candidates returned.")
    print(f"Wrote results to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
