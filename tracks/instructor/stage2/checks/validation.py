"""Feature F3 — external validation score (no remove)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    external_validation_score: float
    has_github: bool


def _non_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def evaluate_external_validation(record: dict) -> ValidationResult:
    signals = record.get("redrob_signals") or {}
    profile = record.get("profile") or {}

    gh = signals.get("github_activity_score", -1)
    has_github = gh != -1

    score = 0.0
    if has_github:
        try:
            score += (float(gh) / 100.0) * 0.6
        except (TypeError, ValueError):
            pass

    pub_keys = ("publications", "patents", "open_source_contributions")
    if any(_non_empty(profile.get(k)) for k in pub_keys):
        score += 0.2

    visibility_keys = ("blog_url", "talks", "portfolio_url")
    if any(_non_empty(profile.get(k)) for k in visibility_keys):
        score += 0.2

    return ValidationResult(
        external_validation_score=round(min(score, 1.0), 4),
        has_github=has_github,
    )
