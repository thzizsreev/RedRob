"""Batch processing and flat FAISS index construction with INSTRUCTOR-large."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np

from pipeline.config import (
    EMPTY_BLOCK_TEXT,
    ID_MAP_FILENAME,
    INDEX_BATCH_SIZE,
    INDEX_FILENAME,
    JD_QUERY_VEC_FILENAME,
    MAX_PASSAGE_TOKENS,
    VECTOR_DIM,
    resolve_passage_prep_workers,
)
from pipeline.extraction import build_candidate_passage, truncate_passage
from pipeline.instructor_encode import (
    INSTRUCTOR,
    batch_size_for_device,
    build_jd_query_vector,
    encode_candidates,
    load_tokenizer,
    log_encode_plan,
)


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


def _prepare_passage(record: dict, tokenizer) -> str:
    passage = build_candidate_passage(record)
    if not passage.strip():
        return EMPTY_BLOCK_TEXT
    return truncate_passage(passage, tokenizer, MAX_PASSAGE_TOKENS)


def _prepare_passages_parallel(
    records: list[dict],
    tokenizer,
    workers: int,
) -> list[str]:
    if workers <= 1 or len(records) <= 1:
        return [_prepare_passage(r, tokenizer) for r in records]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(lambda r: _prepare_passage(r, tokenizer), records)
        )


def _add_vectors_to_index(
    vectors: np.ndarray,
    index: faiss.Index,
    batch_size: int,
) -> None:
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size].astype(np.float32)
        index.add(batch)


def build_vector_index_from_records(
    records: Iterable[dict],
    model: INSTRUCTOR,
    output_dir: Path,
    *,
    device: str,
    limit: int | None = None,
    index_batch_size: int = INDEX_BATCH_SIZE,
    batch_size: int | None = None,
    passage_workers: int | None = None,
) -> tuple[faiss.Index, dict[int, str], np.ndarray]:
    """
    Encode candidates with INSTRUCTOR-large and build a flat FAISS index.

    Returns (index, id_map, jd_query_vector).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    indexed_records: list[tuple[int, dict]] = []
    for idx, record in enumerate(records):
        if limit is not None and idx >= limit:
            break
        indexed_records.append((idx, record))

    if not indexed_records:
        raise ValueError("No candidate records to process")

    encode_batch_size = batch_size_for_device(device, batch_size)
    prep_workers = passage_workers if passage_workers is not None else resolve_passage_prep_workers()

    log_encode_plan(device, encode_batch_size, len(indexed_records), prep_workers)

    tokenizer = load_tokenizer()
    record_list = [r for _, r in indexed_records]
    print(f"Building passages for {len(record_list):,} candidates...")
    passages = _prepare_passages_parallel(record_list, tokenizer, prep_workers)

    print("Encoding candidates (3 instruction passes)...")
    vectors = encode_candidates(model, passages, batch_size=encode_batch_size)

    print("Building JD query vector...")
    jd_query_vec = build_jd_query_vector(model)

    id_map: dict[int, str] = {}
    for faiss_id, (idx, record) in enumerate(indexed_records):
        id_map[faiss_id] = record["candidate_id"]

    index = faiss.IndexFlatIP(VECTOR_DIM)
    _add_vectors_to_index(vectors, index, index_batch_size)

    index_path = output_dir / INDEX_FILENAME
    id_map_path = output_dir / ID_MAP_FILENAME
    jd_query_path = output_dir / JD_QUERY_VEC_FILENAME

    faiss.write_index(index, str(index_path))
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f)
    np.save(jd_query_path, jd_query_vec)

    print(f"Index built: {index.ntotal:,} vectors, dim={VECTOR_DIM}")
    print(f"Saved {index_path}")
    print(f"Saved {id_map_path}")
    print(f"Saved {jd_query_path}")

    return index, id_map, jd_query_vec


def build_vector_index(
    candidates_path: Path,
    model: INSTRUCTOR,
    output_dir: Path,
    *,
    device: str,
    limit: int | None = None,
    index_batch_size: int = INDEX_BATCH_SIZE,
    batch_size: int | None = None,
    passage_workers: int | None = None,
) -> tuple[faiss.Index, dict[int, str], np.ndarray]:
    """Stream candidates from a file, encode vectors, build flat FAISS index."""
    print(f"Processing candidates from {candidates_path}...")
    return build_vector_index_from_records(
        iter_candidates_from_path(candidates_path),
        model,
        output_dir,
        device=device,
        limit=limit,
        index_batch_size=index_batch_size,
        batch_size=batch_size,
        passage_workers=passage_workers,
    )
