"""Neutral-safe behavioral signal accessors for Stage 5."""

from __future__ import annotations

from datetime import date


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        parts = str(value).split("-")
        if len(parts) != 3:
            return None
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError):
        return None


def days_inactive(last_active: str | None, current_date: date) -> int | None:
    active = parse_date(last_active)
    if active is None:
        return None
    return max((current_date - active).days, 0)


def resp_factor(rate: float | None, good_rate: float, floor: float) -> float:
    if rate is None:
        return 1.0
    return clamp(float(rate) / good_rate, floor, 1.0)


def speed_factor(
    hours: float | None,
    slow_hours: float,
    decay_window: float,
    floor: float,
) -> float:
    if hours is None:
        return 1.0
    excess = max(0.0, float(hours) - slow_hours)
    return clamp(1.0 - excess / decay_window, floor, 1.0)


def recency_factor(
    days: int | None,
    fresh_days: int,
    decay_window: int,
    floor: float,
) -> float:
    if days is None:
        return 1.0
    if days <= fresh_days:
        return 1.0
    return clamp(1.0 - (days - fresh_days) / decay_window, floor, 1.0)


def open_factor(open_to_work: bool | None, not_open_factor: float) -> float:
    if open_to_work is None:
        return 1.0
    return 1.0 if open_to_work else not_open_factor


def interview_factor(rate: float | None, floor: float) -> float:
    if rate is None or rate < 0:
        return 1.0
    return clamp(float(rate), floor, 1.0)


def offer_factor(rate: float | None, floor: float) -> float:
    if rate is None or rate < 0:
        return 1.0
    return clamp(float(rate), floor, 1.0)


def market_factor(
    applications_30d: int | None,
    open_to_work: bool | None,
    inactive_factor: float,
) -> float:
    if applications_30d is not None and applications_30d > 0:
        return 1.0
    if open_to_work is True:
        return 1.0
    return inactive_factor
