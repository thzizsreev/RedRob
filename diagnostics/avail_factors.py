"""Availability sub-factor formulas (mirrors tracks/instructor/stage5/signals.py)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class AvailConfig:
    good_response_rate: float = 0.5
    response_floor: float = 0.6
    slow_response_hours: float = 24.0
    response_decay_window_hours: float = 168.0
    speed_floor: float = 0.7
    fresh_days: int = 30
    recency_decay_window: int = 180
    recency_floor: float = 0.6
    not_open_factor: float = 0.85
    interview_floor: float = 0.7
    offer_floor: float = 0.8
    market_inactive_factor: float = 0.95
    avail_min: float = 0.5


FACTOR_NAMES = [
    "resp_factor",
    "speed_factor",
    "recency_factor",
    "open_factor",
    "interview_factor",
    "offer_factor",
    "market_factor",
]

FACTOR_FLOORS = {
    "resp_factor": 0.6,
    "speed_factor": 0.7,
    "recency_factor": 0.6,
    "open_factor": 0.85,
    "interview_factor": 0.7,
    "offer_factor": 0.8,
    "market_factor": 0.95,
}


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


def resp_factor(rate: float | None, cfg: AvailConfig) -> tuple[float, bool]:
    if rate is None:
        return 1.0, True
    return clamp(float(rate) / cfg.good_response_rate, cfg.response_floor, 1.0), False


def speed_factor(hours: float | None, cfg: AvailConfig) -> tuple[float, bool]:
    if hours is None:
        return 1.0, True
    excess = max(0.0, float(hours) - cfg.slow_response_hours)
    return clamp(1.0 - excess / cfg.response_decay_window_hours, cfg.speed_floor, 1.0), False


def recency_factor(days: int | None, cfg: AvailConfig) -> tuple[float, bool]:
    if days is None:
        return 1.0, True
    if days <= cfg.fresh_days:
        return 1.0, False
    return (
        clamp(
            1.0 - (days - cfg.fresh_days) / cfg.recency_decay_window,
            cfg.recency_floor,
            1.0,
        ),
        False,
    )


def open_factor(open_to_work: bool | None, cfg: AvailConfig) -> tuple[float, bool]:
    if open_to_work is None:
        return 1.0, True
    return (1.0 if open_to_work else cfg.not_open_factor), False


def interview_factor(rate: float | None, cfg: AvailConfig) -> tuple[float, bool]:
    if rate is None or rate < 0:
        return 1.0, True
    return clamp(float(rate), cfg.interview_floor, 1.0), False


def offer_factor(rate: float | None, cfg: AvailConfig) -> tuple[float, bool]:
    if rate is None or rate < 0:
        return 1.0, True
    return clamp(float(rate), cfg.offer_floor, 1.0), False


def market_factor(
    applications_30d: int | None,
    open_to_work: bool | None,
    cfg: AvailConfig,
) -> tuple[float, bool]:
    if applications_30d is not None and applications_30d > 0:
        return 1.0, False
    if open_to_work is True:
        return 1.0, False
    return cfg.market_inactive_factor, False


def compute_all_factors(
    signals: dict,
    current_date: date,
    cfg: AvailConfig,
) -> dict[str, float]:
    inactive = days_inactive(signals.get("last_active_date"), current_date)
    rf, _ = resp_factor(signals.get("recruiter_response_rate"), cfg)
    sf, _ = speed_factor(signals.get("avg_response_time_hours"), cfg)
    rcf, _ = recency_factor(inactive, cfg)
    of, _ = open_factor(signals.get("open_to_work_flag"), cfg)
    iff, _ = interview_factor(signals.get("interview_completion_rate"), cfg)
    off, _ = offer_factor(signals.get("offer_acceptance_rate"), cfg)
    mf, _ = market_factor(
        signals.get("applications_submitted_30d"),
        signals.get("open_to_work_flag"),
        cfg,
    )
    return {
        "resp_factor": rf,
        "speed_factor": sf,
        "recency_factor": rcf,
        "open_factor": of,
        "interview_factor": iff,
        "offer_factor": off,
        "market_factor": mf,
    }


def compute_all_factors_with_missing(
    signals: dict,
    current_date: date,
    cfg: AvailConfig,
) -> tuple[dict[str, float], dict[str, bool]]:
    inactive = days_inactive(signals.get("last_active_date"), current_date)
    rf, rf_miss = resp_factor(signals.get("recruiter_response_rate"), cfg)
    sf, sf_miss = speed_factor(signals.get("avg_response_time_hours"), cfg)
    rcf, rcf_miss = recency_factor(inactive, cfg)
    of, of_miss = open_factor(signals.get("open_to_work_flag"), cfg)
    iff, iff_miss = interview_factor(signals.get("interview_completion_rate"), cfg)
    off, off_miss = offer_factor(signals.get("offer_acceptance_rate"), cfg)
    mf, mf_miss = market_factor(
        signals.get("applications_submitted_30d"),
        signals.get("open_to_work_flag"),
        cfg,
    )
    factors = {
        "resp_factor": rf,
        "speed_factor": sf,
        "recency_factor": rcf,
        "open_factor": of,
        "interview_factor": iff,
        "offer_factor": off,
        "market_factor": mf,
    }
    missing = {
        "resp_factor": rf_miss,
        "speed_factor": sf_miss,
        "recency_factor": rcf_miss,
        "open_factor": of_miss,
        "interview_factor": iff_miss,
        "offer_factor": off_miss,
        "market_factor": mf_miss,
    }
    return factors, missing


def availability_multiplier(factors: dict[str, float], cfg: AvailConfig) -> float:
    product = 1.0
    for name in FACTOR_NAMES:
        product *= factors[name]
    return clamp(product, cfg.avail_min, 1.0)
