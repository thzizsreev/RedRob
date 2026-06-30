"""Artifact path resolution for the Q1/Q2 vector test harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.instructor.core.config import (  # noqa: E402
    CANDIDATE_VECTORS_FILENAME,
    ID_MAP_FILENAME,
)
from tracks.shared.paths import RUNTIME_STAGE0_DIR  # noqa: E402

TEST_DIR = Path(__file__).resolve().parent

# Stage 0 precompute output (tracks/instructor/stage0/run.py)
DEFAULT_ARTIFACTS_DIR = RUNTIME_STAGE0_DIR
DEFAULT_VECTORS = DEFAULT_ARTIFACTS_DIR / CANDIDATE_VECTORS_FILENAME
DEFAULT_ID_MAP = DEFAULT_ARTIFACTS_DIR / ID_MAP_FILENAME

DEFAULT_CONFIG = TEST_DIR / "input" / "configs.yaml"
DEFAULT_CANDIDATES = TEST_DIR / "input" / "test_candidates.json"
DEFAULT_SYNTHETIC = TEST_DIR / "input" / "synthetic_cases.yaml"
DEFAULT_OUTPUT = TEST_DIR / "output"


def resolve_vectors_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / CANDIDATE_VECTORS_FILENAME


def resolve_id_map_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / ID_MAP_FILENAME


def load_candidate_ids(id_map_path: Path) -> np.ndarray:
    """Load candidate IDs in FAISS row order from id_map.json."""
    with open(id_map_path, encoding="utf-8") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}
    if not id_map:
        raise ValueError(f"Empty id_map: {id_map_path}")
    n = len(id_map)
    expected = set(range(n))
    if set(id_map.keys()) != expected:
        raise ValueError(f"id_map keys are not contiguous 0..{n - 1}: {id_map_path}")
    return np.array([id_map[i] for i in range(n)], dtype=object)


def load_candidate_matrix(vectors_path: Path) -> np.ndarray:
    return np.load(vectors_path).astype(np.float32)


def validate_artifacts(vectors_path: Path, id_map_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vectors + IDs and verify row alignment."""
    missing = [p for p in (vectors_path, id_map_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifact(s):\n" + "\n".join(f"  {p}" for p in missing)
        )

    matrix = load_candidate_matrix(vectors_path)
    ids = load_candidate_ids(id_map_path)
    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"Vector rows ({matrix.shape[0]}) != id_map entries ({len(ids)}) "
            f"for {vectors_path.parent}"
        )
    return matrix, ids
