"""Ranking-plan skill/date honeypot rules H3, H4, H5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from tracks.instructor.stage2.config import Stage2Config
from tracks.instructor.stage2.honeypot_rules import _parse_date


@dataclass(frozen=True)
class SkillHoneypotEvaluation:
    exclude: bool
    rules_fired: list[str]
    details: dict


def evaluate_skill_honeypot(record: dict, config: Stage2Config) -> SkillHoneypotEvaluation:
    profile = record.get("profile") or {}
    skills = record.get("skills") or []
    career_history = record.get("career_history") or []
    total_years_exp = profile.get("years_of_experience")
    current_date = config.current_date

    rules_fired: list[str] = []
    details: dict = {}

    expert_zero_count = 0
    expert_zero_skills: list[str] = []
    for skill in skills:
        proficiency = str(skill.get("proficiency", "")).lower()
        duration = skill.get("duration_months")
        if proficiency != "expert":
            continue
        try:
            duration_int = 0 if duration is None else int(duration)
        except (TypeError, ValueError):
            duration_int = 0
        if duration_int == 0:
            expert_zero_count += 1
            expert_zero_skills.append(str(skill.get("name", "")))

    if expert_zero_count >= config.expert_zero_threshold:
        rules_fired.append("expert_zero")
        details["expert_zero"] = {
            "count": expert_zero_count,
            "skills": expert_zero_skills[:10],
        }

    if total_years_exp is not None:
        max_skill_years = 0.0
        max_skill_name = ""
        for skill in skills:
            duration = skill.get("duration_months")
            if duration is None:
                continue
            years = int(duration) / 12.0
            if years > max_skill_years:
                max_skill_years = years
                max_skill_name = str(skill.get("name", ""))
        if max_skill_years > float(total_years_exp) + config.skill_years_slack:
            rules_fired.append("skill_years_impossible")
            details["skill_years_impossible"] = {
                "max_skill_years": round(max_skill_years, 2),
                "total_years_exp": total_years_exp,
                "skill": max_skill_name,
            }

    for i, role in enumerate(career_history):
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        if start is not None and end is not None and start > end:
            rules_fired.append("inverted_dates")
            details["inverted_dates"] = {
                "role_index": i,
                "start_date": str(start),
                "end_date": str(end),
            }
            break
        if end is not None and end > current_date:
            rules_fired.append("future_end_date")
            details["future_end_date"] = {"role_index": i, "end_date": str(end)}
            break

    exclude = len(rules_fired) > 0
    return SkillHoneypotEvaluation(exclude=exclude, rules_fired=rules_fired, details=details)
