"""Load precomputed FAISS index vectors and candidate metadata."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np

from tracks.instructor.config import ID_MAP_FILENAME, INDEX_FILENAME, VECTOR_DIM


def _open_candidates(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


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


def load_candidates_json(path: Path) -> list[dict]:
    """Load all candidate records from JSON array, JSONL, or JSONL.gz."""
    return list(iter_candidates_from_path(path))


def load_candidate_ids_from_id_map(
    artifacts_dir: Path,
    *,
    index_filename: str = INDEX_FILENAME,
) -> list[str]:
    """Return candidate IDs in FAISS row order (phase B id alignment)."""
    index, id_map = load_index_and_id_map(artifacts_dir, index_filename=index_filename)
    if index.ntotal != len(id_map):
        raise ValueError(
            f"Index size ({index.ntotal}) does not match id_map ({len(id_map)})"
        )
    return [id_map[faiss_id] for faiss_id in range(index.ntotal)]


def load_index_and_id_map(
    artifacts_dir: Path,
    *,
    index_filename: str = INDEX_FILENAME,
) -> tuple[faiss.Index, dict[int, str]]:
    index_path = artifacts_dir / index_filename
    id_map_path = artifacts_dir / ID_MAP_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. "
            f"Run precompute.py (or python -m tracks.naive.precompute for {index_filename}) first."
        )
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"ID map not found: {id_map_path}. Run precompute.py first."
        )

    index = faiss.read_index(str(index_path))
    with open(id_map_path, encoding="utf-8") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}
    return index, id_map


def load_vectors_from_artifacts(
    artifacts_dir: Path,
    *,
    index_filename: str = INDEX_FILENAME,
    vector_dim: int = VECTOR_DIM,
) -> tuple[list[str], np.ndarray]:
    """Return candidate ids and aligned vectors reconstructed from the FAISS index."""
    index, id_map = load_index_and_id_map(artifacts_dir, index_filename=index_filename)
    if index.ntotal != len(id_map):
        raise ValueError(
            f"Index size ({index.ntotal}) does not match id_map ({len(id_map)})"
        )
    if index.d != vector_dim:
        raise ValueError(f"Expected vector dim {vector_dim}, got {index.d}")

    candidate_ids: list[str] = []
    vectors = np.empty((index.ntotal, vector_dim), dtype=np.float32)

    for faiss_id in range(index.ntotal):
        candidate_id = id_map[faiss_id]
        candidate_ids.append(candidate_id)
        vectors[faiss_id] = index.reconstruct(int(faiss_id))

    return candidate_ids, vectors


def load_jd_query_vector(artifacts_dir: Path) -> np.ndarray:
    from tracks.instructor.config import BLOCK_DIM, JD_QUERY_VEC_FILENAME

    query_path = artifacts_dir / JD_QUERY_VEC_FILENAME
    if not query_path.exists():
        raise FileNotFoundError(
            f"JD query vector not found: {query_path}. Run precompute.py first."
        )
    vec = np.load(query_path).astype(np.float32)
    if vec.shape != (BLOCK_DIM * 3,):
        raise ValueError(f"Expected jd_query_vec shape ({BLOCK_DIM * 3},), got {vec.shape}")
    return vec
