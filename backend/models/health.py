"""Readiness check models."""

from __future__ import annotations

from pydantic import BaseModel


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
    messages: dict[str, str] = {}
