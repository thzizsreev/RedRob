"""Encode and persist Stage 3 Q1/Q2/Q3 query vectors via ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tracks.instructor.core.config import (
    STAGE3_QUERY_MANIFEST_FILENAME,
    STAGE3_QUERY_VECTORS_DIR,
)
from tracks.instructor.core.onnx_embedder import InstructorONNX
from tracks.instructor.stage0.manifest import query_config_hash, write_stage3_query_manifest
from tracks.instructor.stage3.config import load_stage3_config
from tracks.instructor.stage3.query_encode import encode_stage3_queries


def save_query_vectors(
    query_vectors_dir: Path,
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
) -> None:
    query_vectors_dir.mkdir(parents=True, exist_ok=True)
    np.save(query_vectors_dir / "q1_vec.npy", q1.astype(np.float32))
    np.save(query_vectors_dir / "q2_vec.npy", q2.astype(np.float32))
    np.save(query_vectors_dir / "q3_vec.npy", q3.astype(np.float32))
    print(f"Wrote query vectors to {query_vectors_dir}")


def run_stage3_query_precompute(
    model: InstructorONNX,
    output_dir: Path,
    config_path: Path,
) -> Path:
    config = load_stage3_config(config_path)
    query_vectors_dir = output_dir / STAGE3_QUERY_VECTORS_DIR

    q1, q2, q3 = encode_stage3_queries(model, config)
    save_query_vectors(query_vectors_dir, q1, q2, q3)

    rel_dir = STAGE3_QUERY_VECTORS_DIR
    manifest_path = output_dir / STAGE3_QUERY_MANIFEST_FILENAME
    write_stage3_query_manifest(
        manifest_path,
        query_config_hash_value=query_config_hash(config),
        query_vectors_dir=rel_dir,
    )
    return manifest_path
