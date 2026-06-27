"""Health and readiness routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.models.health import ReadyResponse
from backend.services.readiness import check_readiness

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    return check_readiness()
