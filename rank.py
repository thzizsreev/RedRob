#!/usr/bin/env python3
"""
Phase 2 — CPU-only retrieval (5-minute budget).

Loads precomputed flat vector index and JD query vector.
No torch, no INSTRUCTOR, no GPU — FAISS IndexFlatIP search only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from pipeline.config import (
    BLOCK_DIM,
    ID_MAP_FILENAME,
    INDEX_FILENAME,
    JD_QUERY_VEC_FILENAME,
    QUERY_WEIGHTS,
    ROOT_DIR,
)

# --- edit before run ---
ARTIFACTS_PATH = ROOT_DIR / "artifacts" / "sample2"
RESULTS_PATH = ARTIFACTS_PATH / "retrieval_results.json"


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    candidate_id: str
    score: float


def load_index_and_id_map(
    artifacts_dir: Path = ARTIFACTS_PATH,
) -> tuple[faiss.Index, dict[int, str]]:
    index_path = artifacts_dir / INDEX_FILENAME
    id_map_path = artifacts_dir / ID_MAP_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run precompute.py first."
        )
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"ID map not found: {id_map_path}. Run precompute.py first."
        )

    index = faiss.read_index(str(index_path))
    with open(id_map_path, encoding="utf-8") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}
    return index, id_map


def load_jd_query_vector(artifacts_dir: Path = ARTIFACTS_PATH) -> np.ndarray:
    query_path = artifacts_dir / JD_QUERY_VEC_FILENAME
    if not query_path.exists():
        raise FileNotFoundError(
            f"JD query vector not found: {query_path}. Run precompute.py first."
        )
    vec = np.load(query_path).astype(np.float32)
    if vec.shape != (BLOCK_DIM * 3,):
        raise ValueError(f"Expected jd_query_vec shape ({BLOCK_DIM * 3},), got {vec.shape}")
    return vec


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


def retrieve(
    k: int = 300,
    *,
    artifacts_dir: Path = ARTIFACTS_PATH,
    weights: tuple[float, float, float] | None = None,
) -> list[RetrievalHit]:
    """
    Run block-weighted vector retrieval against the precomputed index.

    Returns ranked hits with candidate_id and inner-product score.
    """
    index, id_map = load_index_and_id_map(artifacts_dir)
    query_vector = load_jd_query_vector(artifacts_dir)

    if weights is not None and weights != QUERY_WEIGHTS:
        query_vector = apply_query_weights(query_vector, weights)

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
