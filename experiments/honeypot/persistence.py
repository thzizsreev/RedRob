"""Thread-safe JSONL persistence and idempotency for honeypot results."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from honeypot.config import FAILURES_FILENAME, PASS1_RESULTS_FILENAME, PASS2_RESULTS_FILENAME


class ResultStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._completed: dict[tuple[int, str], dict[str, Any]] = {}
        self._load_existing(PASS1_RESULTS_FILENAME, pass_number=1)
        self._load_existing(PASS2_RESULTS_FILENAME, pass_number=2)

    def _path_for_pass(self, pass_number: int) -> Path:
        if pass_number == 1:
            return self.output_dir / PASS1_RESULTS_FILENAME
        if pass_number == 2:
            return self.output_dir / PASS2_RESULTS_FILENAME
        raise ValueError(f"invalid pass_number: {pass_number}")

    def _load_existing(self, filename: str, *, pass_number: int) -> None:
        path = self.output_dir / filename
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cid = str(row.get("candidate_id", ""))
                if cid:
                    self._completed[(pass_number, cid)] = row

    def has_result(self, pass_number: int, candidate_id: str) -> bool:
        return (pass_number, candidate_id) in self._completed

    def get_result(self, pass_number: int, candidate_id: str) -> dict[str, Any] | None:
        return self._completed.get((pass_number, candidate_id))

    def all_pass1(self) -> list[dict[str, Any]]:
        return [v for (p, _), v in self._completed.items() if p == 1]

    def all_pass2(self) -> list[dict[str, Any]]:
        return [v for (p, _), v in self._completed.items() if p == 2]

    def append_result(self, pass_number: int, row: dict[str, Any]) -> None:
        cid = str(row.get("candidate_id", ""))
        path = self._path_for_pass(pass_number)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if cid:
                self._completed[(pass_number, cid)] = row

    def append_failure(self, row: dict[str, Any]) -> None:
        path = self.output_dir / FAILURES_FILENAME
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
