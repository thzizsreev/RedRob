"""Encode and persist Stage 3 Q1/Q2/Q3 query vectors via ONNX."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.core.config import (  # noqa: E402
    STAGE3_QUERY_MANIFEST_FILENAME,
    STAGE3_QUERY_VECTORS_DIR,
)
from tracks.instructor.core.onnx_embedder import InstructorONNX, load_embedder, unload_embedder  # noqa: E402
from tracks.instructor.stage0.manifest import query_config_hash, write_stage3_query_manifest  # noqa: E402
from tracks.instructor.stage3.config import load_stage3_config  # noqa: E402
from tracks.instructor.stage3.query_encode import encode_stage3_queries  # noqa: E402


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


def main() -> int:
    from tracks.shared.paths import ROOT_DIR, RUNTIME_STAGE0_DIR

    config_path = ROOT_DIR / "config.yaml"
    output_dir = RUNTIME_STAGE0_DIR

    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    model = load_embedder()
    try:
        manifest = run_stage3_query_precompute(model, output_dir, config_path)
        print(f"Done. Manifest: {manifest}")
    finally:
        unload_embedder(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
