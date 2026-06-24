"""Check G — stale coding soft flag (never hard-remove)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tracks.instructor.stage2.checks._history import (
    coding_recency_window,
    iter_roles_sorted,
    months_before,
    normalize_text,
    role_overlaps_window,
)
from tracks.instructor.stage2.config import Stage2Config
from tracks.instructor.stage2.honeypot_rules import _parse_date

RoleCodingClass = Literal["management_only", "hands_on", "ambiguous"]


@dataclass(frozen=True)
class CodingRecencyResult:
    stale_coding: bool
    currently_between_roles: bool
    months_since_last_ic_role: float | None


def _classify_role_coding(title: str, config: Stage2Config) -> RoleCodingClass:
    normalized = normalize_text(title)
    if not normalized:
        return "ambiguous"

    cfg = config.coding_recency
    hands_on = any(signal in normalized for signal in cfg.hands_on_title_signals)
    management = any(signal in normalized for signal in cfg.management_title_signals)

    if hands_on:
        return "hands_on"
    if management:
        return "management_only"
    return "ambiguous"


def _months_since_last_ic_role(
    career_history: list[dict],
    config: Stage2Config,
) -> float | None:
    current_date = config.current_date
    for role in reversed(iter_roles_sorted(career_history)):
        if _classify_role_coding(str(role.get("title", "")), config) != "hands_on":
            continue
        end = _parse_date(role.get("end_date"))
        if end is None:
            return 0.0
        return months_before(current_date, end)
    return None


def evaluate_coding_recency(record: dict, config: Stage2Config) -> CodingRecencyResult:
    career_history = record.get("career_history") or []
    window_start, window_end = coding_recency_window(config)

    overlapping: list[RoleCodingClass] = []
    for role in career_history:
        if role_overlaps_window(role, window_start, window_end, config.current_date):
            overlapping.append(_classify_role_coding(str(role.get("title", "")), config))

    months_since = _months_since_last_ic_role(career_history, config)

    if not overlapping:
        return CodingRecencyResult(
            stale_coding=False,
            currently_between_roles=True,
            months_since_last_ic_role=months_since,
        )

    if any(c == "hands_on" for c in overlapping):
        for role in reversed(iter_roles_sorted(career_history)):
            if _classify_role_coding(str(role.get("title", "")), config) == "hands_on":
                end = _parse_date(role.get("end_date"))
                months_since = 0.0 if end is None else months_before(config.current_date, end)
                break
        return CodingRecencyResult(
            stale_coding=False,
            currently_between_roles=False,
            months_since_last_ic_role=months_since,
        )

    if any(c == "ambiguous" for c in overlapping):
        return CodingRecencyResult(
            stale_coding=False,
            currently_between_roles=False,
            months_since_last_ic_role=months_since,
        )

    return CodingRecencyResult(
        stale_coding=True,
        currently_between_roles=False,
        months_since_last_ic_role=months_since,
    )
