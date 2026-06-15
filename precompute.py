#!/usr/bin/env python3
"""
Offline Phase 1 — block-weighted vector precomputation (no time limits).

Runs the vector encoding pipeline from vector_encoding_plan.md:
  Stage 0: JD anchor vectors  -> artifacts/anchors/*.npy
  Stages 1-7: per-candidate block encoding
  Stage 8: flat vector index  -> artifacts/candidate_index.faiss
                               -> artifacts/id_map.json
"""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from pipeline.anchors import ensure_anchors
from pipeline.config import (
    ANCHORS_DIR,
    ARTIFACTS_DIR,
    MODEL_NAME,
    SAMPLE_CANDIDATES_PATH,
    THRESHOLDS,
)
from pipeline.index import build_vector_index_from_records
from pipeline.model_utils import embedding_dim


def load_encoder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"Loading encoder: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  embedding dimension: {embedding_dim(model)}")
    return model


def load_candidates_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def run_precompute(
    candidates: list[dict],
    model: SentenceTransformer,
    *,
    output_dir: Path = ARTIFACTS_DIR,
    anchors_dir: Path = ANCHORS_DIR,
    rebuild_anchors: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = ensure_anchors(
        model,
        rebuild=rebuild_anchors,
        output_dir=anchors_dir,
    )

    print(f"Processing {len(candidates)} candidates...")
    build_vector_index_from_records(
        candidates,
        model,
        anchors,
        THRESHOLDS,
        output_dir,
    )

    print("Precompute complete.")


def main() -> None:
    candidates = load_candidates_json(SAMPLE_CANDIDATES_PATH)
    model = load_encoder()
    run_precompute(candidates, model)


if __name__ == "__main__":
    main()
