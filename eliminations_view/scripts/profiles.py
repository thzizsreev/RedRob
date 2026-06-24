"""Stream-join candidate profiles from candidates.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def stream_candidates(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(wanted)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = str(record.get("candidate_id", ""))
            if cid in remaining:
                found[cid] = record
                remaining.remove(cid)
    return found


def attach_profile(record: dict[str, Any], profile_row: dict[str, Any] | None) -> dict[str, Any]:
    if not profile_row:
        return record
    record["profile"] = profile_row.get("profile") or {}
    record["career_history"] = profile_row.get("career_history") or []
    record["education"] = profile_row.get("education") or []
    record["skills"] = profile_row.get("skills") or []
    return record
