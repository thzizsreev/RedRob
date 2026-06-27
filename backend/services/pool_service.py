"""Pool lifecycle helpers."""

from __future__ import annotations

import gzip
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile

from backend.services.pool_paths import (
    PoolPaths,
    check_index_readiness,
    create_pool_meta,
    list_pool_ids,
    load_pool_meta,
    new_pool_id,
    resolve_pool,
    save_pool_meta,
)
from backend.settings import Settings


def pool_to_response(paths: PoolPaths) -> dict:
    meta = load_pool_meta(paths)
    checks = check_index_readiness(paths)
    return {
        "pool_id": paths.pool_id,
        "status": meta.get("status", "created"),
        "name": meta.get("name"),
        "description": meta.get("description"),
        "candidate_count": meta.get("candidate_count", 0),
        "indexed": checks.get("indexed", False),
        "created_at": meta.get("created_at"),
        "artifact_checks": checks,
    }


def create_pool(settings: Settings, name: str | None, description: str | None) -> PoolPaths:
    pool_id = new_pool_id()
    paths = resolve_pool(pool_id, settings.api_pools_root)
    paths.ensure_dirs()
    meta = create_pool_meta(name, description)
    save_pool_meta(paths, meta)
    return paths


def get_pool_or_404(settings: Settings, pool_id: str) -> PoolPaths:
    paths = resolve_pool(pool_id, settings.api_pools_root)
    if not paths.meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Pool not found: {pool_id}")
    return paths


def list_pools(settings: Settings) -> list[dict]:
    return [
        pool_to_response(resolve_pool(pool_id, settings.api_pools_root))
        for pool_id in list_pool_ids(settings.api_pools_root)
    ]


def _validate_candidate_line(data: dict, pattern: re.Pattern[str]) -> None:
    cid = data.get("candidate_id")
    if not cid or not pattern.match(str(cid)):
        raise ValueError(f"Invalid candidate_id: {cid!r}")


async def save_candidates_upload(
    paths: PoolPaths,
    upload: UploadFile,
    candidate_id_pattern: str,
) -> int:
    filename = upload.filename or ""
    if not (filename.endswith(".jsonl") or filename.endswith(".jsonl.gz")):
        raise HTTPException(
            status_code=400,
            detail="Upload must be a .jsonl or .jsonl.gz file",
        )

    pattern = re.compile(candidate_id_pattern)
    paths.ensure_dirs()
    dest = paths.candidates_jsonl
    count = 0

    if filename.endswith(".gz"):
        raw = await upload.read()
        try:
            text = gzip.decompress(raw).decode("utf-8")
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid gzip file: {exc}") from exc
        lines = text.splitlines()
    else:
        raw = await upload.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded") from exc
        lines = text.splitlines()

    validated_lines: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON line: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Each line must be a JSON object")
        try:
            _validate_candidate_line(data, pattern)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        validated_lines.append(json.dumps(data, ensure_ascii=False))
        count += 1

    if count == 0:
        raise HTTPException(status_code=400, detail="No candidate records found in upload")

    dest.write_text("\n".join(validated_lines) + "\n", encoding="utf-8")

    meta = load_pool_meta(paths)
    meta["candidate_count"] = count
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta["status"] = "candidates_uploaded"
    meta["indexed"] = False
    save_pool_meta(paths, meta)
    return count


def mark_pool_indexed(paths: PoolPaths) -> None:
    meta = load_pool_meta(paths)
    meta["indexed"] = True
    meta["status"] = "indexed"
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_pool_meta(paths, meta)


def clear_stage_artifacts(paths: PoolPaths) -> None:
    for stage_dir in (paths.stage0, paths.stage1, paths.stage2, paths.stage3, paths.stage4, paths.stage5):
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
