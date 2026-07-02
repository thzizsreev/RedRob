"""TRACER honeypot detection — high-precision hard reject only."""

from __future__ import annotations

from ranker.jd_config import AI_SKILL_PATTERN, TITLE_NEGATIVE, TITLE_TIER1, TITLE_TIER2


def is_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """Hard reject only when confidence is very high (~official honeypots)."""
    flags: list[str] = []
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    yoe = float(profile.get("years_of_experience", 0))
    title = profile.get("current_title", "")

    if TITLE_NEGATIVE.search(title):
        flags.append("negative_title")

    expert_zero = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 4:
        flags.append("expert_zero_duration")

    total_months = sum(h.get("duration_months", 0) for h in history)
    if yoe > 0 and total_months > yoe * 12 + 18:
        flags.append("timeline_impossible")

    expert_ai = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and AI_SKILL_PATTERN.search(s.get("name", ""))
    )
    if yoe < 2 and expert_ai >= 6:
        flags.append("yoe_skill_impossible")

    return len(flags) > 0, flags


def soft_trap_flags(candidate: dict) -> list[str]:
    """Soft penalties applied in feature layer and TRM, not hard reject."""
    flags: list[str] = []
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])

    if len(history) >= 3:
        descs = [h.get("description", "").strip().lower() for h in history if h.get("description")]
        if len(descs) >= 3:
            unique_ratio = len(set(descs)) / len(descs)
            if unique_ratio <= 0.34:
                flags.append("career_desc_reuse")

    title = profile.get("current_title", "")
    ai_title = bool(TITLE_TIER1.search(title) or TITLE_TIER2.search(title))
    ai_skill_count = sum(1 for s in skills if AI_SKILL_PATTERN.search(s.get("name", "")))
    if not ai_title and ai_skill_count >= 8:
        weak = sum(
            1
            for s in skills
            if AI_SKILL_PATTERN.search(s.get("name", ""))
            and (s.get("duration_months", 0) < 6 or s.get("endorsements", 0) < 2)
        )
        if weak >= ai_skill_count * 0.6:
            flags.append("keyword_stuffer")

    return flags
