"""Stage 8: Batch processing and flat vector index construction."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.candidate import process_one_candidate
from pipeline.config import INDEX_BATCH_SIZE
from pipeline.model_utils import embedding_dim


def _open_candidates(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates_from_path(path: Path) -> Iterable[dict]:
    """Yield candidate records from JSONL(.gz) or a JSON array file."""
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array in {path}")
        yield from records
        return

    with _open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_vector_index_from_records(
    records: Iterable[dict],
    model: SentenceTransformer,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float],
    output_dir: Path,
    *,
    limit: int | None = None,
    batch_size: int = INDEX_BATCH_SIZE,
) -> tuple[faiss.Index, dict[int, str]]:
    """Encode candidate records and build a flat FAISS index (O(N) exhaustive search)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dim = embedding_dim(model) * 3

    index = faiss.IndexFlatIP(dim)

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

    for record in records:
        if limit is not None and faiss_id >= limit:
            break

        candidate_id, vec = process_one_candidate(record, model, anchors, thresholds)

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
    """Stream candidates from a file, encode vectors, build flat FAISS index."""
    print(f"Processing candidates from {candidates_path}...")
    return build_vector_index_from_records(
        iter_candidates_from_path(candidates_path),
        model,
        anchors,
        thresholds,
        output_dir,
        limit=limit,
        batch_size=batch_size,
    )
