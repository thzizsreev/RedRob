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
from time import perf_counter

from sentence_transformers import SentenceTransformer

from pipeline.anchors import ensure_anchors
from pipeline.config import (
    ANCHORS_DIR,
    ARTIFACTS_DIR,
    CANDIDATES_JSONL_PATH,
    MODEL_NAME,
    MODEL_RAM_GB_ESTIMATE,
    SAMPLE_CANDIDATES_PATH,
    THRESHOLDS,
)
from pipeline.index import build_vector_index, build_vector_index_from_records
from pipeline.model_utils import embedding_dim, load_sentence_transformer, resolve_device
from pipeline.parallel import resolve_workers


def load_encoder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"Loading encoder: {model_name}")
    model = load_sentence_transformer(model_name)
    print(f"  embedding dimension: {embedding_dim(model)}")
    return model


def load_candidates_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def run_precompute(
    candidates_path: Path,
    model: SentenceTransformer,
    *,
    output_dir: Path = ARTIFACTS_DIR,
    anchors_dir: Path = ANCHORS_DIR,
    rebuild_anchors: bool = False,
    workers: int | None = None,
    limit: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = ensure_anchors(
        model,
        rebuild=rebuild_anchors,
        output_dir=anchors_dir,
    )

    worker_count = resolve_workers(workers)
    device = resolve_device()
    if device == "cuda":
        print("Estimated VRAM: ~1 model on GPU (single encode worker)")
    else:
        est_ram_gb = worker_count * MODEL_RAM_GB_ESTIMATE
        print(f"Estimated model RAM: ~{est_ram_gb:.1f}GB ({worker_count} thread-local model(s))")

    if candidates_path.suffix == ".json":
        candidates = load_candidates_json(candidates_path)
        print(f"Processing {len(candidates)} candidates from {candidates_path}...")
        build_vector_index_from_records(
            candidates,
            model,
            anchors,
            THRESHOLDS,
            output_dir,
            limit=limit,
            workers=worker_count,
        )
    else:
        build_vector_index(
            candidates_path,
            model,
            anchors,
            THRESHOLDS,
            output_dir,
            limit=limit,
            workers=worker_count,
        )

    print("Precompute complete.")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def main() -> None:
    candidates_path = SAMPLE_CANDIDATES_PATH
    print(f"Using candidates file: {candidates_path}")
    started = perf_counter()
    model = load_encoder()
    run_precompute(candidates_path, model)
    elapsed = perf_counter() - started
    print(f"Precompute complete in {_format_duration(elapsed)}.")


def main_full() -> None:
    candidates_path = CANDIDATES_JSONL_PATH
    print(f"Using candidates file: {candidates_path}")

    started = perf_counter()
    model = load_encoder()
    run_precompute(candidates_path, model)
    elapsed = perf_counter() - started

    print(f"Precompute complete in {_format_duration(elapsed)}.")


if __name__ == "__main__":
    main()
