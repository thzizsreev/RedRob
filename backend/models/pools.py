"""Pool API models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CreatePoolRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class PoolResponse(BaseModel):
    pool_id: str
    status: str
    name: str | None = None
    description: str | None = None
    candidate_count: int = 0
    indexed: bool = False
    created_at: datetime | None = None
    artifact_checks: dict[str, bool] = Field(default_factory=dict)


class PoolListResponse(BaseModel):
    pools: list[PoolResponse]


class UploadCandidatesResponse(BaseModel):
    pool_id: str
    candidate_count: int
    message: str = "Candidates uploaded successfully"


class IndexJobResponse(BaseModel):
    job_id: str
    pool_id: str
    status: str = "queued"
