"""Load Stage 0 FAISS index and candidate vectors."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from tracks.instructor.core.config import CANDIDATE_VECTORS_FILENAME, VECTOR_DIM
from tracks.instructor.core.io import load_index_and_id_map, load_vectors_from_artifacts
from tracks.shared.paths import ROOT_DIR


@dataclass(frozen=True)
class RetrievalAssets:
    index: faiss.Index
    vectors: np.ndarray
    id_to_row: dict[str, int]
    row_to_id: list[str]


def resolve_stage0_dir(stage0_pointer_path: Path) -> Path:
    with open(stage0_pointer_path, encoding="utf-8") as f:
        pointer = json.load(f)

    stage0_rel = pointer.get("stage0_dir", "artifacts/runtime/stage0")
    stage0_dir = Path(stage0_rel)
    if not stage0_dir.is_absolute():
        stage0_dir = (ROOT_DIR / stage0_dir).resolve()

    if not stage0_dir.is_dir():
        raise FileNotFoundError(f"Stage 0 directory not found: {stage0_dir}")
    return stage0_dir


def load_retrieval_assets(stage0_dir: Path) -> RetrievalAssets:
    index, id_map = load_index_and_id_map(stage0_dir)

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(
            f"FAISS index must use METRIC_INNER_PRODUCT, got {index.metric_type}"
        )

    row_to_id = [id_map[i] for i in range(index.ntotal)]
    id_to_row = {cid: i for i, cid in enumerate(row_to_id)}

    vectors_path = stage0_dir / CANDIDATE_VECTORS_FILENAME
    if vectors_path.exists():
        vectors = np.load(vectors_path).astype(np.float32)
        print(f"Loaded candidate vectors: {vectors.shape}")
    else:
        warnings.warn(
            f"{vectors_path} not found — reconstructing vectors from FAISS index",
            stacklevel=2,
        )
        _, vectors = load_vectors_from_artifacts(stage0_dir)

    if vectors.shape != (index.ntotal, VECTOR_DIM):
        raise ValueError(
            f"Vector shape {vectors.shape} does not match index "
            f"({index.ntotal}, {VECTOR_DIM})"
        )

    print(f"FAISS index: {index.ntotal:,} vectors, dim={index.d}")
    return RetrievalAssets(
        index=index,
        vectors=vectors,
        id_to_row=id_to_row,
        row_to_id=row_to_id,
    )


def build_survivor_row_indices(
    stage2_df,
    id_to_row: dict[str, int],
) -> np.ndarray:
    unknown: list[str] = []
    indices: list[int] = []
    for cid in stage2_df["candidate_id"].to_list():
        row = id_to_row.get(str(cid))
        if row is None:
            unknown.append(str(cid))
        else:
            indices.append(row)

    if unknown:
        examples = unknown[:5]
        raise ValueError(
            f"{len(unknown)} Stage 2 survivor(s) not found in id_map. Examples: {examples}"
        )

    return np.array(indices, dtype=np.int64)
