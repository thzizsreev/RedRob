"""Pool-scoped artifact path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tracks.instructor.core.config import (
    CANDIDATE_FEATURES_FILENAME,
    CANDIDATE_VECTORS_FILENAME,
    ID_MAP_FILENAME,
    INDEX_FILENAME,
    STAGE1_CLUSTER_LABELS_FILENAME,
    STAGE1_CLUSTER_MANIFEST_FILENAME,
)

CANDIDATES_FILENAME = "candidates.jsonl"
META_FILENAME = "pool_meta.json"


@dataclass(frozen=True)
class PoolPaths:
    pool_id: str
    root: Path

    @property
    def candidates_jsonl(self) -> Path:
        return self.root / CANDIDATES_FILENAME

    @property
    def meta_path(self) -> Path:
        return self.root / META_FILENAME

    @property
    def stage0(self) -> Path:
        return self.root / "stage0"

    @property
    def stage1(self) -> Path:
        return self.root / "stage1"

    @property
    def stage2(self) -> Path:
        return self.root / "stage2"

    @property
    def stage3(self) -> Path:
        return self.root / "stage3"

    @property
    def stage4(self) -> Path:
        return self.root / "stage4"

    @property
    def stage5(self) -> Path:
        return self.root / "stage5"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.stage0,
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
            self.jobs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def new_pool_id() -> str:
    return uuid4().hex[:12]


def resolve_pool(pool_id: str, pools_root: Path) -> PoolPaths:
    return PoolPaths(pool_id=pool_id, root=pools_root / pool_id)


def list_pool_ids(pools_root: Path) -> list[str]:
    if not pools_root.exists():
        return []
    return sorted(
        p.name for p in pools_root.iterdir() if p.is_dir() and (p / META_FILENAME).exists()
    )


def load_pool_meta(paths: PoolPaths) -> dict:
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"Pool not found: {paths.pool_id}")
    return json.loads(paths.meta_path.read_text(encoding="utf-8"))


def save_pool_meta(paths: PoolPaths, meta: dict) -> None:
    paths.ensure_dirs()
    paths.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def create_pool_meta(name: str | None, description: str | None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "name": name,
        "description": description,
        "created_at": now,
        "updated_at": now,
        "candidate_count": 0,
        "indexed": False,
        "status": "created",
    }


def check_index_readiness(paths: PoolPaths) -> dict[str, bool]:
    stage0_ok = all(
        (paths.stage0 / f).exists()
        for f in (
            INDEX_FILENAME,
            ID_MAP_FILENAME,
            CANDIDATE_VECTORS_FILENAME,
            CANDIDATE_FEATURES_FILENAME,
        )
    )
    stage1_ok = all(
        (paths.stage1 / f).exists()
        for f in (STAGE1_CLUSTER_LABELS_FILENAME, STAGE1_CLUSTER_MANIFEST_FILENAME)
    )
    candidates_ok = paths.candidates_jsonl.exists() and paths.candidates_jsonl.stat().st_size > 0
    return {
        "candidates_uploaded": candidates_ok,
        "stage0_artifacts": stage0_ok,
        "stage1_cluster_artifacts": stage1_ok,
        "indexed": stage0_ok and stage1_ok and candidates_ok,
    }
