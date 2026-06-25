"""Ranking-plan skill/date honeypot rules H3, H4 (skill_years_ceiling), H5."""

from __future__ import annotations

from dataclasses import dataclass

from tracks.instructor.stage2.config import HoneypotConfig, Stage2Config
from tracks.instructor.stage2.honeypot_rules import _parse_date


@dataclass(frozen=True)
class SkillHoneypotEvaluation:
    exclude: bool
    rules_fired: list[str]
    details: dict


def _skill_years_ceiling(yoe: float, hp: HoneypotConfig) -> float:
    return max(
        yoe * hp.max_skill_overshoot_factor,
        yoe + hp.early_career_skill_buffer_years,
    )


def _rule_skill_years_ceiling(
    skills: list[dict],
    total_years_exp: float | None,
    hp: HoneypotConfig,
) -> tuple[bool, dict | None]:
    worst: dict | None = None
    worst_overshoot = 0.0

    for skill in skills:
        duration = skill.get("duration_months")
        if duration is None:
            continue
        try:
            skill_years = int(duration) / 12.0
        except (TypeError, ValueError):
            continue
        skill_name = str(skill.get("name", ""))

        if skill_years > hp.max_skill_years_absolute:
            overshoot = skill_years - hp.max_skill_years_absolute
            if overshoot > worst_overshoot:
                worst_overshoot = overshoot
                worst = {
                    "skill": skill_name,
                    "skill_years": round(skill_years, 2),
                    "total_years_exp": total_years_exp,
                    "ceiling_years": None,
                    "violation": "absolute",
                }
            continue

        if total_years_exp is not None:
            ceiling = _skill_years_ceiling(float(total_years_exp), hp)
            if skill_years > ceiling:
                overshoot = skill_years - ceiling
                if overshoot > worst_overshoot:
                    worst_overshoot = overshoot
                    worst = {
                        "skill": skill_name,
                        "skill_years": round(skill_years, 2),
                        "total_years_exp": total_years_exp,
                        "ceiling_years": round(ceiling, 2),
                        "violation": "ceiling",
                    }

    return worst is not None, worst


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

    # H4 skill_years_ceiling — tiered ceiling vs YOE + absolute cap.
    # Replaces retired skill_years_impossible (max skill > YOE + slack).
    fired, detail = _rule_skill_years_ceiling(
        skills, total_years_exp, config.honeypot
    )
    if fired:
        rules_fired.append("skill_years_ceiling")
        if detail:
            details["skill_years_ceiling"] = detail

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
