"""Stage 8: Batch processing and flat vector index construction."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.candidate import process_one_candidate
from pipeline.config import INDEX_BATCH_SIZE, MODEL_NAME
from pipeline.model_utils import embedding_dim, resolve_device
from pipeline.parallel import encode_candidate_task, resolve_workers


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


def _encode_sequential(
    indexed_records: list[tuple[int, dict]],
    model: SentenceTransformer,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[tuple[int, str, np.ndarray]]:
    results: list[tuple[int, str, np.ndarray]] = []
    for idx, record in indexed_records:
        candidate_id, vec = process_one_candidate(record, model, anchors, thresholds)
        results.append((idx, candidate_id, vec))
        if (idx + 1) % 5000 == 0:
            print(f"  processed {idx + 1:,} candidates")
    return results


def _encode_parallel(
    indexed_records: list[tuple[int, dict]],
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float],
    workers: int,
    model_name: str = MODEL_NAME,
) -> list[tuple[int, str, np.ndarray]]:
    tasks = [
        (idx, record, anchors, thresholds, model_name)
        for idx, record in indexed_records
    ]
    results: list[tuple[int, str, np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, candidate_id, vec in executor.map(encode_candidate_task, tasks):
            results.append((idx, candidate_id, vec))
            if (idx + 1) % 5000 == 0:
                print(f"  processed {idx + 1:,} candidates")
    results.sort(key=lambda item: item[0])
    return results


def _add_results_to_index(
    results: list[tuple[int, str, np.ndarray]],
    index: faiss.Index,
    batch_size: int,
) -> dict[int, str]:
    id_map: dict[int, str] = {}
    current_batch: list[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal current_batch
        if not current_batch:
            return
        arr = np.array(current_batch, dtype=np.float32)
        index.add(arr)
        current_batch = []

    for faiss_id, candidate_id, vec in results:
        id_map[faiss_id] = candidate_id
        current_batch.append(vec)
        if len(current_batch) >= batch_size:
            flush_batch()

    flush_batch()
    return id_map


def build_vector_index_from_records(
    records: Iterable[dict],
    model: SentenceTransformer,
    anchors: dict[str, np.ndarray],
    thresholds: dict[str, float],
    output_dir: Path,
    *,
    limit: int | None = None,
    batch_size: int = INDEX_BATCH_SIZE,
    workers: int | None = None,
) -> tuple[faiss.Index, dict[int, str]]:
    """
    Encode candidate records and build a flat FAISS index (O(N) exhaustive search).

    The model is used for sequential mode (workers=1) only; parallel workers load
    thread-local models via pipeline.parallel.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dim = embedding_dim(model) * 3
    worker_count = resolve_workers(workers)
    device = resolve_device()

    indexed_records: list[tuple[int, dict]] = []
    for idx, record in enumerate(records):
        if limit is not None and idx >= limit:
            break
        indexed_records.append((idx, record))

    print(
        f"Encoding {len(indexed_records):,} candidates with {worker_count} worker(s) "
        f"on {device}..."
    )

    if worker_count == 1:
        encoded = _encode_sequential(indexed_records, model, anchors, thresholds)
    else:
        encoded = _encode_parallel(
            indexed_records, anchors, thresholds, worker_count, MODEL_NAME
        )

    index = faiss.IndexFlatIP(dim)
    id_map = _add_results_to_index(encoded, index, batch_size)

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
    workers: int | None = None,
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
        workers=workers,
    )
