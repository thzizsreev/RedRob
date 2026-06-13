#!/usr/bin/env python3
"""
Offline Phase 1 — block-weighted vector precomputation (no time limits).

Runs the vector encoding pipeline from vector_encoding_plan.md:
  Stage 0: JD anchor vectors  -> artifacts/anchors/*.npy
  Stages 1-7: per-candidate block encoding
  Stage 8: FAISS HNSW index   -> artifacts/candidate_index.faiss
                               -> artifacts/id_map.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from pipeline.anchors import ensure_anchors
from pipeline.config import (
    ANCHORS_DIR,
    ARTIFACTS_DIR,
    DEFAULT_CANDIDATES_PATH,
    MODEL_NAME,
    THRESHOLDS,
)
from pipeline.index import build_vector_index
from pipeline.model_utils import embedding_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build block-weighted candidate vector index (offline precompute)."
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory for FAISS index and id_map.json",
    )
    parser.add_argument(
        "--anchors-dir",
        type=Path,
        default=ANCHORS_DIR,
        help="Directory for anchor .npy files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N candidates (for testing)",
    )
    parser.add_argument(
        "--rebuild-anchors",
        action="store_true",
        help="Rebuild anchor vectors even if they already exist",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Sentence-transformers model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.candidates_path.exists():
        print(f"Error: candidates file not found: {args.candidates_path}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading encoder: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"  embedding dimension: {embedding_dim(model)}")

    anchors = ensure_anchors(
        model,
        rebuild=args.rebuild_anchors,
        output_dir=args.anchors_dir,
    )

    build_vector_index(
        candidates_path=args.candidates_path,
        model=model,
        anchors=anchors,
        thresholds=THRESHOLDS,
        output_dir=args.output_dir,
        limit=args.limit,
    )

    print("Precompute complete.")


if __name__ == "__main__":
    main()
