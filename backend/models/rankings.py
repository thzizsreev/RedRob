"""Ranking API models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateRankingRequest(BaseModel):
    pool_id: str
    config_path: str | None = None
    random_seed: int | None = None


class RankingJobResponse(BaseModel):
    job_id: str
    pool_id: str
    status: str = "queued"


class RankingResultItem(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str


class RankingResultsResponse(BaseModel):
    job_id: str
    pool_id: str
    items: list[RankingResultItem]
    summary: dict[str, Any] = Field(default_factory=dict)
