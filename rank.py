#!/usr/bin/env python3
"""
Phase 2 — CPU-only retrieval (5-minute budget).

Loads precomputed flat vector index and runs block-weighted query retrieval.
Downstream LightGBM ranking and CSV output can be layered on top.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import ARTIFACTS_DIR, MODEL_NAME, QUERY_WEIGHTS
from pipeline.query import build_query_vector, build_query_vector_from_text


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    candidate_id: str
    score: float


def load_encoder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def load_index_and_id_map(
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[faiss.Index, dict[int, str]]:
    index_path = artifacts_dir / "candidate_index.faiss"
    id_map_path = artifacts_dir / "id_map.json"

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
    artifacts_dir: Path = ARTIFACTS_DIR,
    model_name: str = MODEL_NAME,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
    model: SentenceTransformer | None = None,
) -> list[RetrievalHit]:
    """
    Run block-weighted vector retrieval against the precomputed index.

    Returns ranked hits with candidate_id and inner-product score.
    """
    index, id_map = load_index_and_id_map(artifacts_dir)

    if model is None:
        model = load_encoder(model_name)

    query_vector = build_query_vector(model, weights=weights)
    scores, indices = search_index(index, query_vector, k)
    return hits_from_search(scores, indices, id_map)


def retrieve_from_text(
    query_text: str,
    k: int = 10,
    *,
    artifacts_dir: Path = ARTIFACTS_DIR,
    model_name: str = MODEL_NAME,
    weights: tuple[float, float, float] = QUERY_WEIGHTS,
    model: SentenceTransformer | None = None,
) -> list[RetrievalHit]:
    """Retrieve top-k candidates for a free-text query."""
    index, id_map = load_index_and_id_map(artifacts_dir)

    if model is None:
        model = load_encoder(model_name)

    query_vector = build_query_vector_from_text(model, query_text, weights=weights)
    scores, indices = search_index(index, query_vector, k)
    return hits_from_search(scores, indices, id_map)


def write_results_json(results: list[RetrievalHit], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"rank": hit.rank, "candidate_id": hit.candidate_id, "score": hit.score}
        for hit in results
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> list[RetrievalHit]:
    output_path = ARTIFACTS_DIR / "retrieval_results.json"
    results = retrieve(k=300)
    write_results_json(results, output_path)

    for hit in results[:10]:
        print(f"  {hit.rank:3d}. {hit.candidate_id}  score={hit.score:.4f}")
    print(f"Retrieval complete: {len(results)} candidates returned.")
    print(f"Wrote results to {output_path}")
    return results


if __name__ == "__main__":
    main()
