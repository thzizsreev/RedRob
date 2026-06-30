"""Load candidate records from JSONL, JSONL.gz, or JSON array."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path


def _open_candidates(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_candidates(path: Path) -> Iterable[dict]:
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


def extract_signals(record: dict) -> dict:
    signals = record.get("redrob_signals") or {}
    return {
        "candidate_id": str(record.get("candidate_id", "")),
        "last_active_date": signals.get("last_active_date"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "applications_submitted_30d": signals.get("applications_submitted_30d"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "avg_response_time_hours": signals.get("avg_response_time_hours"),
        "interview_completion_rate": signals.get("interview_completion_rate"),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate"),
    }


def load_signals_for_ids(path: Path, candidate_ids: set[str]) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    wanted = set(candidate_ids)
    found: dict[str, dict] = {}
    for record in iter_candidates(path):
        cid = str(record.get("candidate_id", ""))
        if cid not in wanted or cid in found:
            continue
        found[cid] = extract_signals(record)
        if len(found) == len(wanted):
            break

    missing = wanted - set(found.keys())
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} candidate(s) missing from {path}. Examples: {examples}"
        )
    return found
