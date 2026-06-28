"""Deterministic template + resume clause composition (Phase 1 decode)."""

from __future__ import annotations

from constants import (
    resume_behav_text,
    resume_career_text,
    resume_tech_text,
)

HIGH_THRESHOLD = 0.65
LOW_THRESHOLD = 0.35

_RESUME_BY_DIMENSION = {
    "tech": resume_tech_text.strip(),
    "career": resume_career_text.strip(),
    "behav": resume_behav_text.strip(),
}


def _direction_clause(score: float, resume_detail: str) -> str:
    detail = resume_detail.rstrip(".")
    if score >= HIGH_THRESHOLD:
        return f"Backed by concrete profile evidence: {detail}"
    if score < LOW_THRESHOLD:
        return (
            f"However, engagement signals are weak: {detail}, "
            "which may limit recruiter reach"
        )
    return f"Profile notes {detail}, though further confirmation may help"


def compose_clause(dimension: str, score: float) -> str:
    """Return resume-specific continuation clause for one dimension."""
    resume_detail = _RESUME_BY_DIMENSION[dimension]
    return _direction_clause(score, resume_detail)
