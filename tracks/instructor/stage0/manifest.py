"""Stage 3 query-vector manifest for config invalidation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tracks.instructor.stage3.config import Stage3Config


@dataclass(frozen=True)
class Stage3QueryManifest:
    version: int
    created_at: str
    query_config_hash: str
    query_vectors_dir: str


def query_config_hash(config: Stage3Config) -> str:
    payload: dict = {
        "q2_text": config.q2_text,
        "q3_text": config.q3_text,
        "subspace_weights_q1": config.subspace_weights_q1.as_tuple(),
        "subspace_weights_q2": config.subspace_weights_q2.as_tuple(),
        "subspace_weights_q3": config.subspace_weights_q3.as_tuple(),
    }
    if config.q1_facets:
        payload["q1_facets"] = {f.id: f.text for f in config.q1_facets}
        payload["q1_facet_weights"] = {f.id: f.weight for f in config.q1_facets}
    else:
        payload["q1_text"] = config.q1_text
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_stage3_query_manifest(
    manifest_path: Path,
    *,
    query_config_hash_value: str,
    query_vectors_dir: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "created_at": utc_now_iso(),
        "query_config_hash": query_config_hash_value,
        "query_vectors_dir": query_vectors_dir,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


def load_stage3_query_manifest(manifest_path: Path) -> Stage3QueryManifest:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing Stage 3 query manifest: {manifest_path}. "
            "Re-run tracks/instructor/stage0/stage3_query_precompute.py"
        )
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    return Stage3QueryManifest(
        version=int(raw["version"]),
        created_at=str(raw["created_at"]),
        query_config_hash=str(raw["query_config_hash"]),
        query_vectors_dir=str(raw["query_vectors_dir"]),
    )


def verify_query_manifest(
    manifest_path: Path,
    config_path: Path,
    artifacts_path: Path,
) -> Path:
    """Load manifest, verify hash matches current config, return query vectors dir."""
    from tracks.instructor.stage3.config import load_stage3_config

    manifest = load_stage3_query_manifest(manifest_path)
    config = load_stage3_config(config_path)
    expected = query_config_hash(config)
    if manifest.query_config_hash != expected:
        raise ValueError(
            "Stage 3 query config changed since last Stage 0 precompute. "
            f"manifest hash={manifest.query_config_hash[:12]}… "
            f"current hash={expected[:12]}…. "
            "Re-run tracks/instructor/stage0/stage3_query_precompute.py"
        )
    vectors_dir = Path(manifest.query_vectors_dir)
    if not vectors_dir.is_absolute():
        vectors_dir = (artifacts_path / vectors_dir).resolve()
    return vectors_dir
