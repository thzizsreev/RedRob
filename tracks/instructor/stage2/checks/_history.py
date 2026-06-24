"""Shared career-history helpers for Stage 2 JD checks."""

from __future__ import annotations

from datetime import date

from tracks.instructor.stage2.config import Stage2Config
from tracks.instructor.stage2.honeypot_rules import _parse_date


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return str(value).lower().strip()


def unique_employers(career_history: list[dict]) -> list[str]:
    seen: set[str] = set()
    employers: list[str] = []
    for role in career_history:
        company = normalize_text(role.get("company"))
        if company and company not in seen:
            seen.add(company)
            employers.append(company)
    return employers


def iter_roles_sorted(career_history: list[dict]) -> list[dict]:
    roles = list(career_history)

    def sort_key(role: dict) -> tuple[date, date]:
        start = _parse_date(role.get("start_date")) or date.min
        end = _parse_date(role.get("end_date")) or date.max
        return (start, end)

    return sorted(roles, key=sort_key)


def role_overlaps_window(
    role: dict,
    window_start: date,
    window_end: date,
    current_date: date,
) -> bool:
    start = _parse_date(role.get("start_date"))
    if start is None:
        return False
    end = _parse_date(role.get("end_date")) or current_date
    return start <= window_end and end >= window_start


def months_before(end: date, start: date) -> float:
    return (end.year - start.year) * 12 + (end.month - start.month)


def subtract_months(d: date, months: int) -> date:
    month_index = d.month - 1 - months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, min(d.day, 28))


def coding_recency_window(config: Stage2Config) -> tuple[date, date]:
    end = config.current_date
    start = subtract_months(end, config.coding_recency.stale_coding_window_months)
    return start, end


def role_duration_years(role: dict) -> float | None:
    months = role.get("duration_months")
    if months is not None:
        try:
            return float(months) / 12.0
        except (TypeError, ValueError):
            pass
    start = _parse_date(role.get("start_date"))
    end = _parse_date(role.get("end_date"))
    if start is None:
        return None
    if end is None:
        return None
    delta = end - start
    return max(delta.days / 365.25, 0.0)
