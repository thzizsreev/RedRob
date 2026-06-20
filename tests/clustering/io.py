"""Load candidate records and precomputed FAISS vectors (test adapter)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracks.instructor.config import INDEX_FILENAME, VECTOR_DIM
from tracks.instructor.io import (
    load_candidates_json,
    load_vectors_from_artifacts,
)


@dataclass(frozen=True)
class ClusteringInputs:
    candidate_ids: list[str]
    records: list[dict]
    vectors: np.ndarray


def load_vectors_and_records(
    candidates_path: Path,
    artifacts_path: Path,
    *,
    index_filename: str = INDEX_FILENAME,
    vector_dim: int = VECTOR_DIM,
) -> ClusteringInputs:
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

    return ClusteringInputs(
        candidate_ids=candidate_ids,
        records=aligned_records,
        vectors=vectors,
    )
