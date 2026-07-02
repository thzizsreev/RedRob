"""Behavioral signal multiplier from redrob_signals."""

from __future__ import annotations

from datetime import datetime, timezone


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def behavioral_multiplier(signals: dict) -> float:
    """Return multiplier in roughly [0.5, 1.15]."""
    mult = 1.0

    rr = signals.get("recruiter_response_rate")
    if rr is not None:
        if rr >= 0.6:
            mult += 0.08
        elif rr >= 0.35:
            mult += 0.03
        elif rr < 0.15:
            mult -= 0.12

    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days_ago = (datetime(2026, 6, 12, tzinfo=timezone.utc) - last_active).days
        if days_ago <= 30:
            mult += 0.05
        elif days_ago <= 90:
            mult += 0.02
        elif days_ago > 180:
            mult -= 0.15

    if signals.get("open_to_work_flag"):
        mult += 0.04

    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        mult += 0.04
    elif saved >= 2:
        mult += 0.02

    icr = signals.get("interview_completion_rate")
    if icr is not None and icr >= 0.7:
        mult += 0.03
    elif icr is not None and icr < 0.4:
        mult -= 0.05

    notice = signals.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            mult += 0.04
        elif notice > 60:
            mult -= 0.06

    if signals.get("verified_email") and signals.get("verified_phone"):
        mult += 0.02

    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= 80:
        mult += 0.02
    elif completeness < 40:
        mult -= 0.05

    return max(0.45, min(1.18, mult))
