#!/usr/bin/env python3
"""
Phase 2 — CPU-only retrieval script (5-minute budget).

Loads precomputed FAISS index and runs block-weighted query retrieval.
Downstream LightGBM ranking and CSV output can be layered on top.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss

from pipeline.config import (
    ARTIFACTS_DIR,
    FAISS_EF_SEARCH,
    MODEL_NAME,
    QUERY_WEIGHTS,
)
from pipeline.query import build_query_vector


def _load_encoder(model_name: str, use_onnx: bool):
    if use_onnx:
        try:
            from sentence_transformers import SentenceTransformer

            print("Note: ONNX runtime path requested; falling back to sentence-transformers.")
        except ImportError:
            pass

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run block-weighted FAISS retrieval.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory containing candidate_index.faiss and id_map.json",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=300,
        help="Number of candidates to retrieve",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=FAISS_EF_SEARCH,
        help="HNSW efSearch parameter",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Encoder model (must match precompute)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write retrieval results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_path = args.artifacts_dir / "candidate_index.faiss"
    id_map_path = args.artifacts_dir / "id_map.json"

    if not index_path.exists():
        print(f"Error: index not found: {index_path}", file=sys.stderr)
        print("Run precompute.py first.", file=sys.stderr)
        sys.exit(1)
    if not id_map_path.exists():
        print(f"Error: id map not found: {id_map_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    index.hnsw.efSearch = args.ef_search

    with open(id_map_path, encoding="utf-8") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}

    print(f"Loading encoder: {args.model}")
    model = _load_encoder(args.model, use_onnx=True)

    query_vector = build_query_vector(model, weights=QUERY_WEIGHTS)
    query_matrix = query_vector.reshape(1, -1)

    print(f"Searching top-{args.k} (efSearch={args.ef_search})...")
    scores, indices = index.search(query_matrix, args.k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        candidate_id = id_map.get(int(idx), f"UNKNOWN_{idx}")
        results.append({"rank": rank, "candidate_id": candidate_id, "score": float(score)})
        if rank <= 10:
            print(f"  {rank:3d}. {candidate_id}  score={score:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {len(results)} results to {args.output}")

    print(f"Retrieval complete: {len(results)} candidates returned.")


if __name__ == "__main__":
    main()
