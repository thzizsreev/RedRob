"""Stream candidates from JSONL or gzipped JSONL."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Iterator


def _open_candidates(path: Path):
    path_str = str(path)
    if path.suffix == ".gz" or path_str.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".json":
        return None
    return open(path, "r", encoding="utf-8")


def load_candidates(
    path: Path,
    *,
    limit: int | None = None,
) -> Iterator[dict]:
    """
    Yield candidate records from JSONL, JSONL.gz, or JSON array file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    count = 0

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array in {path}")
        for record in records:
            if limit is not None and count >= limit:
                break
            if isinstance(record, dict) and record.get("candidate_id"):
                yield record
                count += 1
        return

    handle = _open_candidates(path)
    if handle is None:
        raise ValueError(f"Unsupported file format: {path}")

    with handle as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed line {line_no}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(record, dict) or not record.get("candidate_id"):
                print(
                    f"Warning: skipping line {line_no}: missing candidate_id",
                    file=sys.stderr,
                )
                continue
            yield record
            count += 1


def load_all_candidates(path: Path, *, limit: int | None = None) -> list[dict]:
    return list(load_candidates(path, limit=limit))
