"""Shared API models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobType(str, Enum):
    index = "index"
    ranking = "ranking"


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None


class StageTimingResponse(BaseModel):
    stage: int
    label: str
    elapsed_seconds: float


class JobResponse(BaseModel):
    job_id: str
    type: JobType
    status: JobStatus
    pool_id: str
    progress: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    timings: list[StageTimingResponse] | None = None
