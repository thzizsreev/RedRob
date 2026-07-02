"""Evidence-linked reasoning strings for TRACER submission CSV."""

from __future__ import annotations

from ranker.evidence import best_career_phrase, trusted_skills_with_duration


def build_reasoning(rank: int, candidate: dict, context: dict) -> str:
    """Generate 1-2 sentence justification from actual profile fields."""
    f = context["features"]
    sem = context.get("semantic_score")
    title = f["current_title"]
    yoe = f["years_of_experience"]
    skills = f.get("matched_skills") or []
    rr = f.get("recruiter_response_rate")
    notice = f.get("notice_period_days")
    country = f.get("country", "")
    location = f.get("location", "")
    evidence = f.get("career_evidence") or []
    penalties = f.get("penalty_reasons") or []
    trap_risk = context.get("trap_risk", 0.0)

    parts: list[str] = []

    if rank <= 15:
        parts.append(f"{title} with {yoe:.1f} yrs")
    else:
        parts.append(f"{title}; {yoe:.1f} yrs experience")

    if sem is not None and sem >= 0.55:
        parts.append(f"TRACER semantic {sem:.2f}")
    elif sem is not None and rank <= 40:
        parts.append(f"moderate semantic match ({sem:.2f})")

    career_phrase = best_career_phrase(candidate)
    if career_phrase:
        parts.append(f"career: '{career_phrase}'")
    elif evidence:
        parts.append(f"career shows {evidence[0]}")
    elif f.get("career_score", 0) >= 0.4:
        parts.append("production ML/search signals in career history")

    skill_labels = trusted_skills_with_duration(candidate, limit=3)
    if skill_labels:
        parts.append("skills: " + ", ".join(skill_labels))
    elif skills:
        parts.append(f"strong on {', '.join(skills[:3])}")

    if rr is not None:
        parts.append(f"response rate {rr:.2f}")

    concerns: list[str] = []
    if notice is not None and notice > 60:
        concerns.append(f"{notice}-day notice")
    if rr is not None and rr < 0.15:
        concerns.append("low recruiter responsiveness")
    if country and country.lower() != "india" and rank <= 30:
        concerns.append(f"based in {location or country}")
    if "consulting_only" in penalties:
        concerns.append("consulting-heavy background")
    if trap_risk >= 0.15:
        concerns.append("minor trap-risk flags reviewed")

    if concerns:
        parts.append("concern: " + "; ".join(concerns[:2]))
    elif rank >= 80:
        parts.append("adjacent fit for Senior AI Engineer role")

    if (
        not context.get("honeypot")
        and rank <= 10
        and (f.get("title_score", 0) >= 0.75 or (sem or 0) >= 0.6)
    ):
        parts.append("TRACER hybrid+rerank fit for ranking/retrieval JD")

    text = "; ".join(parts)
    if len(text) > 220:
        text = text[:217] + "..."
    return text
