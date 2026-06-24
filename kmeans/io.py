"""Load candidate records and precomputed FAISS vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracks.instructor.core.config import INDEX_FILENAME, VECTOR_DIM
from tracks.instructor.core.io import (
    load_candidates_json,
    load_index_and_id_map,
    load_vectors_from_artifacts,
)


@dataclass(frozen=True)
class KMeansInputs:
    candidate_ids: list[str]
    records: list[dict]
    vectors: np.ndarray


def load_inputs(
    candidates_path: Path,
    artifacts_path: Path,
    *,
    index_filename: str = INDEX_FILENAME,
    vector_dim: int = VECTOR_DIM,
) -> KMeansInputs:
    records = load_candidates_json(candidates_path)
    by_id = {record["candidate_id"]: record for record in records}

    candidate_ids, vectors = load_vectors_from_artifacts(
        artifacts_path,
        index_filename=index_filename,
        vector_dim=vector_dim,
    )

    aligned_records: list[dict] = []
    for candidate_id in candidate_ids:
        if candidate_id not in by_id:
            raise ValueError(
                f"Candidate {candidate_id} is in the index but missing from {candidates_path}"
            )
        aligned_records.append(by_id[candidate_id])

    return KMeansInputs(
        candidate_ids=candidate_ids,
        records=aligned_records,
        vectors=vectors,
    )


def load_records_for_ids(
    candidates_path: Path,
    candidate_ids: list[str],
) -> list[dict]:
    """Align candidate records to a fixed ID list from precompute."""
    records = load_candidates_json(candidates_path)
    by_id = {record["candidate_id"]: record for record in records}

    aligned_records: list[dict] = []
    for candidate_id in candidate_ids:
        if candidate_id not in by_id:
            raise ValueError(
                f"Candidate {candidate_id} is in precompute but missing from {candidates_path}"
            )
        aligned_records.append(by_id[candidate_id])

    return aligned_records


def load_vectors_for_candidate_ids(
    artifacts_path: Path,
    candidate_ids: list[str],
    *,
    index_filename: str = INDEX_FILENAME,
    vector_dim: int = VECTOR_DIM,
) -> np.ndarray:
    """Return (n, vector_dim) vectors aligned to candidate_ids order."""
    index, id_map = load_index_and_id_map(
        artifacts_path,
        index_filename=index_filename,
    )
    if index.d != vector_dim:
        raise ValueError(f"Expected vector dim {vector_dim}, got {index.d}")

    id_to_faiss_id = {candidate_id: faiss_id for faiss_id, candidate_id in id_map.items()}
    vectors = np.empty((len(candidate_ids), vector_dim), dtype=np.float32)

    for row, candidate_id in enumerate(candidate_ids):
        if candidate_id not in id_to_faiss_id:
            raise ValueError(
                f"Candidate {candidate_id} not found in FAISS index at {artifacts_path}"
            )
        vectors[row] = index.reconstruct(int(id_to_faiss_id[candidate_id]))

    return vectors


def load_anchor_vector(artifacts_path: Path) -> np.ndarray:
    from tracks.instructor.core.io import load_jd_query_vector

    return load_jd_query_vector(artifacts_path)
