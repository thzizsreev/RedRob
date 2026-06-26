"""Read/write precompute artifacts and manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tracks.shared.paths import ROOT_DIR


@dataclass(frozen=True)
class PrecomputeManifest:
    version: int
    created_at: str
    query_config_hash: str
    cohort_config_hash: str
    cohort_row_count: int
    paths: dict[str, str]

    @property
    def query_vectors_dir(self) -> Path:
        return _resolve_manifest_path(self.paths["query_vectors_dir"])

    @property
    def cohort_stage2(self) -> Path:
        return _resolve_manifest_path(self.paths["cohort_stage2"])

    @property
    def cohort_features(self) -> Path:
        return _resolve_manifest_path(self.paths["cohort_features"])

    @property
    def survivor_row_indices(self) -> Path:
        return _resolve_manifest_path(self.paths["survivor_row_indices"])

    @property
    def stage0_pointer(self) -> Path:
        return _resolve_manifest_path(self.paths["stage0_pointer"])


def _resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def load_manifest(manifest_path: Path) -> PrecomputeManifest:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing precompute manifest: {manifest_path}. "
            "Run: python test_stage_3/precompute/run.py"
        )
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    return PrecomputeManifest(
        version=int(raw["version"]),
        created_at=str(raw["created_at"]),
        query_config_hash=str(raw["query_config_hash"]),
        cohort_config_hash=str(raw["cohort_config_hash"]),
        cohort_row_count=int(raw["cohort_row_count"]),
        paths=dict(raw["paths"]),
    )


def write_manifest(manifest_path: Path, manifest: PrecomputeManifest) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": manifest.version,
        "created_at": manifest.created_at,
        "query_config_hash": manifest.query_config_hash,
        "cohort_config_hash": manifest.cohort_config_hash,
        "cohort_row_count": manifest.cohort_row_count,
        "paths": manifest.paths,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_query_vectors(query_vectors_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = np.load(query_vectors_dir / "q1_vec.npy").astype(np.float32)
    q2 = np.load(query_vectors_dir / "q2_vec.npy").astype(np.float32)
    q3 = np.load(query_vectors_dir / "q3_vec.npy").astype(np.float32)
    return q1, q2, q3


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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
