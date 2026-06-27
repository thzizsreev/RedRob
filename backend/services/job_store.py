"""Thread-safe in-memory job registry."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.models.common import JobStatus, JobType


@dataclass
class JobRecord:
    job_id: str
    type: JobType
    pool_id: str
    status: JobStatus = JobStatus.queued
    progress: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    config_path: str | None = None
    random_seed: int | None = None


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._gpu_lock = threading.Lock()

    def create_job(
        self,
        *,
        job_type: JobType,
        pool_id: str,
        config_path: str | None = None,
        random_seed: int | None = None,
    ) -> JobRecord:
        job_id = uuid4().hex
        record = JobRecord(
            job_id=job_id,
            type=job_type,
            pool_id=pool_id,
            config_path=config_path,
            random_seed=random_seed,
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: str | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = progress
            if error is not None:
                record.error = error
            if result is not None:
                record.result = result
            if started:
                record.started_at = datetime.now(timezone.utc)
            if finished:
                record.finished_at = datetime.now(timezone.utc)
            return record

    def acquire_gpu_lock(self, timeout: float | None = None) -> bool:
        if timeout is None:
            return self._gpu_lock.acquire(blocking=True)
        return self._gpu_lock.acquire(blocking=True, timeout=timeout)

    def release_gpu_lock(self) -> None:
        if self._gpu_lock.locked():
            self._gpu_lock.release()


_store: InMemoryJobStore | None = None


def get_job_store() -> InMemoryJobStore:
    global _store
    if _store is None:
        _store = InMemoryJobStore()
    return _store
