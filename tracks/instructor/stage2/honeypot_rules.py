"""Deterministic timeline honeypot rules (inline at Stage 2 gate time)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from tracks.instructor.stage2.config import HoneypotConfig, Stage2Config


@dataclass(frozen=True)
class HoneypotEvaluation:
    exclude: bool
    rules_fired: list[str]
    details: dict


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = [
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ]
    day = min(d.day, days_in_month[month - 1])
    return date(year, month, day)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        parts = str(value).split("-")
        if len(parts) != 3:
            return None
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError):
        return None


def _rule_future_start(
    career_history: list[dict],
    current_date: date,
) -> tuple[bool, dict | None]:
    for i, role in enumerate(career_history):
        start = _parse_date(role.get("start_date"))
        if start is not None and start > current_date:
            return True, {"role_index": i, "start_date": str(start)}
    return False, None


def _rule_duration_overshoot(
    career_history: list[dict],
    current_date: date,
    grace_days: int,
) -> tuple[bool, dict | None]:
    grace_end = current_date + timedelta(days=grace_days)
    for i, role in enumerate(career_history):
        start = _parse_date(role.get("start_date"))
        duration_months = role.get("duration_months")
        if start is None or duration_months is None:
            continue
        try:
            months = int(duration_months)
        except (TypeError, ValueError):
            continue
        implied_end = _add_months(start, months)
        if implied_end > grace_end:
            return True, {
                "role_index": i,
                "start_date": str(start),
                "duration_months": months,
                "implied_end": str(implied_end),
            }
    return False, None


def _rule_role_overlap(
    career_history: list[dict],
    current_date: date,
) -> tuple[bool, dict | None]:
    roles: list[tuple[int, date, date]] = []
    for i, role in enumerate(career_history):
        start = _parse_date(role.get("start_date"))
        if start is None:
            continue
        end = _parse_date(role.get("end_date"))
        if end is None:
            end = current_date
        roles.append((i, start, end))

    roles.sort(key=lambda x: x[1])
    for j in range(len(roles) - 1):
        idx_a, _start_a, end_a = roles[j]
        idx_b, start_b, _end_b = roles[j + 1]
        if start_b < end_a:
            return True, {
                "role_index_a": idx_a,
                "role_index_b": idx_b,
                "overlap_start": str(start_b),
                "overlap_end": str(end_a),
            }
    return False, None


def _rule_timeline_sum(
    career_history: list[dict],
    claimed_years: float | None,
    tolerance_years: float,
) -> tuple[bool, dict | None]:
    if claimed_years is None:
        return False, None
    total_months = 0
    for role in career_history:
        duration = role.get("duration_months")
        if duration is None:
            continue
        try:
            total_months += int(duration)
        except (TypeError, ValueError):
            continue
    total_years = total_months / 12.0
    if total_years > claimed_years + tolerance_years:
        return True, {
            "total_years_from_roles": round(total_years, 2),
            "claimed_years": claimed_years,
            "tolerance_years": tolerance_years,
        }
    return False, None


def _rule_graduation_vs_exp(
    education: list[dict],
    claimed_years: float | None,
    current_date: date,
    buffer_years: int,
) -> tuple[bool, dict | None]:
    if claimed_years is None:
        return False, None
    grad_years = [
        int(edu["end_year"])
        for edu in education
        if edu.get("end_year") is not None
    ]
    if not grad_years:
        return False, None
    latest_grad = max(grad_years)
    max_possible = current_date.year - latest_grad + buffer_years
    if claimed_years > max_possible:
        return True, {
            "latest_graduation_year": latest_grad,
            "max_possible_experience": max_possible,
            "claimed_years": claimed_years,
        }
    return False, None


def _should_exclude_timeline(
    hard_rules: list[str],
    soft_rules: list[str],
) -> bool:
    if hard_rules:
        return True
    if "timeline_sum" in soft_rules and "graduation_vs_exp" in soft_rules:
        return True
    return False


def evaluate_timeline_honeypot(
    record: dict,
    config: Stage2Config,
) -> HoneypotEvaluation:
    """Evaluate R1–R5 timeline rules per honeypot_filter_plan."""
    hp: HoneypotConfig = config.honeypot
    current_date = config.current_date
    career_history = record.get("career_history") or []
    education = record.get("education") or []
    profile = record.get("profile") or {}
    claimed_years = profile.get("years_of_experience")

    rules_fired: list[str] = []
    details: dict = {}
    hard_rules: list[str] = []
    soft_rules: list[str] = []

    fired, detail = _rule_future_start(career_history, current_date)
    if fired:
        rules_fired.append("future_start")
        hard_rules.append("future_start")
        if detail:
            details["future_start"] = detail

    fired, detail = _rule_duration_overshoot(
        career_history, current_date, hp.duration_overshoot_grace_days
    )
    if fired:
        rules_fired.append("duration_overshoot")
        hard_rules.append("duration_overshoot")
        if detail:
            details["duration_overshoot"] = detail

    fired, detail = _rule_role_overlap(career_history, current_date)
    if fired:
        rules_fired.append("role_overlap")
        hard_rules.append("role_overlap")
        if detail:
            details["role_overlap"] = detail

    fired, detail = _rule_timeline_sum(
        career_history, claimed_years, hp.experience_overage_tolerance_years
    )
    if fired:
        rules_fired.append("timeline_sum")
        soft_rules.append("timeline_sum")
        if detail:
            details["timeline_sum"] = detail

    fired, detail = _rule_graduation_vs_exp(
        education, claimed_years, current_date, hp.grad_to_work_buffer_years
    )
    if fired:
        rules_fired.append("graduation_vs_exp")
        soft_rules.append("graduation_vs_exp")
        if detail:
            details["graduation_vs_exp"] = detail

    exclude = _should_exclude_timeline(hard_rules, soft_rules)
    return HoneypotEvaluation(exclude=exclude, rules_fired=rules_fired, details=details)
