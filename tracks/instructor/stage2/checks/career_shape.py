"""Feature F1 — career shape signals (no remove)."""

from __future__ import annotations

from dataclasses import dataclass

from tracks.instructor.stage2.checks._history import (
    iter_roles_sorted,
    normalize_text,
    role_duration_years,
    unique_employers,
)
from tracks.instructor.stage2.config import Stage2Config

_SENIORITY_RANKS = (
    ("intern", 0),
    ("engineer", 1),
    ("senior", 2),
    ("staff", 3),
    ("principal", 4),
)


@dataclass(frozen=True)
class CareerShapeResult:
    avg_tenure_per_employer: float
    short_hop_count: int
    title_progression_jumps: int


def _seniority_rank(title: str) -> int | None:
    normalized = normalize_text(title)
    if not normalized:
        return None
    best_rank = None
    for token, rank in _SENIORITY_RANKS:
        if token in normalized:
            if best_rank is None or rank > best_rank:
                best_rank = rank
    return best_rank


def evaluate_career_shape(record: dict, config: Stage2Config) -> CareerShapeResult:
    career_history = record.get("career_history") or []
    threshold = config.career_shape.short_hop_threshold_years

    employer_durations: dict[str, float] = {}
    short_hop_count = 0

    for role in career_history:
        company = normalize_text(role.get("company")) or "unknown"
        duration = role_duration_years(role)
        if duration is None:
            continue
        employer_durations[company] = employer_durations.get(company, 0.0) + duration
        if duration < threshold:
            short_hop_count += 1

    employers = unique_employers(career_history)
    if employer_durations:
        avg_tenure = sum(employer_durations.values()) / len(employer_durations)
    elif employers:
        avg_tenure = 0.0
    else:
        avg_tenure = 0.0

    progression_jumps = 0
    sorted_roles = iter_roles_sorted(career_history)
    prev_employer: str | None = None
    prev_rank: int | None = None

    for role in sorted_roles:
        employer = normalize_text(role.get("company"))
        rank = _seniority_rank(str(role.get("title", "")))
        if prev_employer is not None and employer != prev_employer:
            if rank is not None and prev_rank is not None and rank > prev_rank:
                progression_jumps += 1
        prev_employer = employer
        if rank is not None:
            prev_rank = rank

    return CareerShapeResult(
        avg_tenure_per_employer=round(avg_tenure, 3),
        short_hop_count=short_hop_count,
        title_progression_jumps=progression_jumps,
    )
