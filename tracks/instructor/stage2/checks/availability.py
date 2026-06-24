"""Check D — availability soft flags (never hard-remove)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from tracks.instructor.stage2.config import Stage2Config
from tracks.instructor.stage2.honeypot_rules import _parse_date


@dataclass(frozen=True)
class AvailabilityFlags:
    stale_profile: bool
    low_responder: bool
    not_open: bool
    notice_period_days: int | None


def evaluate_availability(record: dict, config: Stage2Config) -> AvailabilityFlags:
    signals = record.get("redrob_signals") or {}
    current_date = config.current_date

    last_active = _parse_date(signals.get("last_active_date"))
    stale_cutoff = current_date - timedelta(days=config.stale_days)
    stale_profile = last_active is not None and last_active < stale_cutoff

    response_rate = signals.get("recruiter_response_rate")
    low_responder = (
        response_rate is not None and float(response_rate) < config.min_response_rate
    )

    open_flag = signals.get("open_to_work_flag")
    not_open = open_flag is False

    notice_raw = signals.get("notice_period_days")
    notice_period_days: int | None = None
    if notice_raw is not None:
        try:
            notice_period_days = int(notice_raw)
        except (TypeError, ValueError):
            notice_period_days = None

    return AvailabilityFlags(
        stale_profile=stale_profile,
        low_responder=low_responder,
        not_open=not_open,
        notice_period_days=notice_period_days,
    )
