"""Per-candidate feature extraction."""

from __future__ import annotations

import re

from ranker import jd_config as jd

_LANGCHAIN_ONLY = re.compile(
    r"langchain|openai api|anthropic api", re.I
)
_CV_ONLY = re.compile(
    r"\b(computer vision|speech recognition|robotics|object detection|image classification)\b",
    re.I
)
_NLP_IR = re.compile(
    r"\b(nlp|retrieval|ranking|embedding|llm|recsys|recommendation)\b",
    re.I
)


def _norm(s: str) -> str:
    return s.lower().strip()


def _title_score(title: str) -> float:
    if jd.TITLE_TIER1.search(title):
        return 1.0
    if jd.TITLE_TIER2.search(title):
        return 0.75
    if jd.TITLE_NEGATIVE.search(title):
        return 0.05
    if re.search(r"\b(engineer|developer|scientist)\b", title, re.I):
        return 0.35
    return 0.15


def _skill_trust(skill: dict) -> float:
    prof = jd.PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.25)
    months = skill.get("duration_months", 0)
    endorse = skill.get("endorsements", 0)
    duration_factor = min(1.0, months / 24.0)
    endorse_factor = min(1.0, endorse / 20.0)
    return prof * (0.5 + 0.3 * duration_factor + 0.2 * endorse_factor)


def _skill_score(skills: list[dict]) -> tuple[float, list[str]]:
    must_hits: list[tuple[str, float]] = []
    nice_hits: list[tuple[str, float]] = []
    for skill in skills:
        name = _norm(skill.get("name", ""))
        trust = _skill_trust(skill)
        for term in jd.MUST_HAVE_SKILLS:
            if term in name or name in term:
                must_hits.append((skill["name"], trust))
                break
        else:
            for term in jd.NICE_TO_HAVE_SKILLS:
                if term in name or name in term:
                    nice_hits.append((skill["name"], trust))
                    break

    must_score = min(1.0, sum(t for _, t in must_hits) / 4.0) if must_hits else 0.0
    nice_score = min(1.0, sum(t for _, t in nice_hits) / 3.0) if nice_hits else 0.0
    combined = 0.65 * must_score + 0.35 * nice_score

    top_names = sorted(
        [(n, t) for n, t in must_hits + nice_hits],
        key=lambda x: x[1],
        reverse=True,
    )
    matched = [n for n, _ in top_names[:4]]
    return combined, matched


def _career_score(history: list[dict], summary: str) -> tuple[float, list[str]]:
    text_parts = [summary] + [
        f"{h.get('title', '')} {h.get('description', '')}" for h in history
    ]
    blob = _norm(" ".join(text_parts))

    pos = sum(1 for kw in jd.CAREER_POSITIVE if kw in blob)
    neg = sum(1 for kw in jd.CAREER_NEGATIVE if kw in blob)
    product = sum(1 for kw in jd.PRODUCT_INDICATORS if kw in blob)

    score = min(1.0, pos * 0.12 + product * 0.08)
    score = max(0.0, score - neg * 0.15)

    evidence = [kw for kw in jd.CAREER_POSITIVE if kw in blob][:3]
    return score, evidence


def _experience_score(yoe: float) -> float:
    if jd.IDEAL_YOE_MIN <= yoe <= jd.IDEAL_YOE_MAX:
        return 1.0
    if jd.SOFT_YOE_MIN <= yoe < jd.IDEAL_YOE_MIN:
        return 0.6 + 0.4 * (yoe - jd.SOFT_YOE_MIN) / (jd.IDEAL_YOE_MIN - jd.SOFT_YOE_MIN)
    if jd.IDEAL_YOE_MAX < yoe <= jd.SOFT_YOE_MAX:
        return 0.6 + 0.4 * (jd.SOFT_YOE_MAX - yoe) / (jd.SOFT_YOE_MAX - jd.IDEAL_YOE_MAX)
    if yoe >= 3.0:
        return 0.35
    return 0.15


def _location_score(profile: dict, signals: dict) -> float:
    country = _norm(profile.get("country", ""))
    location = _norm(profile.get("location", ""))
    score = 0.2
    if country == "india":
        if any(c in location for c in jd.PREFERRED_CITIES):
            score = 1.0
        else:
            score = 0.75
        if signals.get("open_to_work_flag"):
            score = min(1.0, score + 0.05)
    elif signals.get("willing_to_relocate"):
        score = 0.45
    return score


def _assessment_score(signals: dict, matched_skills: list[str]) -> float:
    assessments = signals.get("skill_assessment_scores") or {}
    if not assessments:
        return 0.5
    relevant = []
    matched_lower = {_norm(s) for s in matched_skills}
    for skill_name, score in assessments.items():
        sn = _norm(skill_name)
        if any(sn in m or m in sn for m in matched_lower) or jd.AI_SKILL_PATTERN.search(skill_name):
            relevant.append(score / 100.0)
    if not relevant:
        relevant = [v / 100.0 for v in assessments.values()]
    return sum(relevant) / len(relevant) if relevant else 0.5


def _anti_pattern_penalty(candidate: dict) -> tuple[float, list[str]]:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    summary = profile.get("summary", "")
    blob = _norm(summary + " " + " ".join(s.get("name", "") for s in skills))

    penalties: list[str] = []
    penalty = 0.0

    companies = [h.get("company", "") for h in history] + [profile.get("current_company", "")]
    consulting_hits = sum(
        1 for c in companies if any(cf in _norm(c) for cf in jd.CONSULTING_FIRMS)
    )
    if consulting_hits >= len(companies) and len(companies) >= 2:
        penalty += 0.25
        penalties.append("consulting_only")

    if len(history) >= 4:
        short_stints = sum(1 for h in history if h.get("duration_months", 0) < 18)
        if short_stints >= len(history) * 0.7:
            penalty += 0.15
            penalties.append("title_chaser")

    if _LANGCHAIN_ONLY.search(blob) and not re.search(
        r"production|shipped|deployed|ndcg|retrieval|ranking", blob, re.I
    ):
        penalty += 0.2
        penalties.append("langchain_only")

    if _CV_ONLY.search(blob) and not _NLP_IR.search(blob):
        penalty += 0.15
        penalties.append("cv_without_nlp")

    from ranker.honeypot import soft_trap_flags

    for flag in soft_trap_flags(candidate):
        if flag == "career_desc_reuse":
            penalty += 0.10
            penalties.append("career_desc_reuse")
        elif flag == "keyword_stuffer":
            penalty += 0.12
            penalties.append("keyword_stuffer")

    return min(0.5, penalty), penalties


def extract_features(candidate: dict) -> dict:
    """Extract normalized feature dict and metadata for reasoning."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])

    title = profile.get("current_title", "")
    yoe = float(profile.get("years_of_experience", 0))

    skill_score, matched_skills = _skill_score(skills)
    career_score, career_evidence = _career_score(history, profile.get("summary", ""))
    penalty, penalty_reasons = _anti_pattern_penalty(candidate)

    github = float(signals.get("github_activity_score", 0) or 0)
    if github >= 50 and matched_skills:
        career_score = min(1.0, career_score + 0.03)

    for edu in candidate.get("education", [])[:2]:
        field = _norm(edu.get("field_of_study", ""))
        if any(k in field for k in ("computer", "machine learning", "artificial intelligence", "data science")):
            career_score = min(1.0, career_score + 0.02)
            break

    return {
        "title_score": _title_score(title),
        "career_score": career_score,
        "skill_score": skill_score,
        "experience_score": _experience_score(yoe),
        "location_score": _location_score(profile, signals),
        "assessment_score": _assessment_score(signals, matched_skills),
        "anti_pattern_penalty": penalty,
        "current_title": title,
        "years_of_experience": yoe,
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "matched_skills": matched_skills,
        "career_evidence": career_evidence,
        "penalty_reasons": penalty_reasons,
        "notice_period_days": signals.get("notice_period_days"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "open_to_work": signals.get("open_to_work_flag", False),
    }
