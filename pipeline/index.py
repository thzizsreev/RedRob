"""Stage 8: Batch processing and FAISS HNSW index construction."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.candidate import process_one_candidate
from pipeline.config import (
    FAISS_EF_CONSTRUCTION,
    FAISS_HNSW_M,
    INDEX_BATCH_SIZE,
)
from pipeline.model_utils import embedding_dim


def _open_candidates(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def build_vector_index(
    candidates_path: Path,
    model: SentenceTransformer,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float],
    output_dir: Path,
    *,
    limit: int | None = None,
    batch_size: int = INDEX_BATCH_SIZE,
) -> tuple[faiss.Index, dict[int, str]]:
    """
    Stream candidates from JSONL(.gz), encode vectors, build FAISS HNSW index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dim = embedding_dim(model) * 3

    index = faiss.IndexHNSWFlat(dim, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = FAISS_EF_CONSTRUCTION

    id_map: dict[int, str] = {}
    current_batch: list[np.ndarray] = []
    faiss_id = 0

    def flush_batch() -> None:
        nonlocal current_batch
        if not current_batch:
            return
        arr = np.array(current_batch, dtype=np.float32)
        index.add(arr)
        current_batch = []

    print(f"Processing candidates from {candidates_path}...")
    with _open_candidates(candidates_path) as f:
        for line in f:
            if limit is not None and faiss_id >= limit:
                break

            record = json.loads(line.strip())
            candidate_id, vec = process_one_candidate(
                record, model, anchors, thresholds
            )

            id_map[faiss_id] = candidate_id
            current_batch.append(vec)

            if len(current_batch) >= batch_size:
                flush_batch()

            faiss_id += 1
            if faiss_id % 5000 == 0:
                print(f"  processed {faiss_id:,} candidates")

    flush_batch()

    index_path = output_dir / "candidate_index.faiss"
    id_map_path = output_dir / "id_map.json"

    faiss.write_index(index, str(index_path))
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f)

    print(f"Index built: {index.ntotal:,} vectors, dim={dim}")
    print(f"Saved {index_path}")
    print(f"Saved {id_map_path}")

    return index, id_map
