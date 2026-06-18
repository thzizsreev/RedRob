"""Load candidate records and precomputed FAISS vectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from pipeline.config import ID_MAP_FILENAME, INDEX_FILENAME, VECTOR_DIM


@dataclass(frozen=True)
class ClusteringInputs:
    candidate_ids: list[str]
    records: list[dict]
    vectors: np.ndarray


def load_candidates_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def load_index_and_id_map(artifacts_dir: Path) -> tuple[faiss.Index, dict[int, str]]:
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


def load_vectors_and_records(
    candidates_path: Path,
    artifacts_path: Path,
) -> ClusteringInputs:
    records = load_candidates_json(candidates_path)
    by_id = {record["candidate_id"]: record for record in records}

    index, id_map = load_index_and_id_map(artifacts_path)
    if index.ntotal != len(id_map):
        raise ValueError(
            f"Index size ({index.ntotal}) does not match id_map ({len(id_map)})"
        )
    if index.d != VECTOR_DIM:
        raise ValueError(f"Expected vector dim {VECTOR_DIM}, got {index.d}")

    candidate_ids: list[str] = []
    aligned_records: list[dict] = []
    vectors = np.empty((index.ntotal, VECTOR_DIM), dtype=np.float32)

    for faiss_id in range(index.ntotal):
        candidate_id = id_map[faiss_id]
        if candidate_id not in by_id:
            raise ValueError(
                f"Candidate {candidate_id} is in the index but missing from {candidates_path}"
            )
        candidate_ids.append(candidate_id)
        aligned_records.append(by_id[candidate_id])
        vectors[faiss_id] = index.reconstruct(int(faiss_id))

    return ClusteringInputs(
        candidate_ids=candidate_ids,
        records=aligned_records,
        vectors=vectors,
    )
