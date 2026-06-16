"""Stage 1: Role-contextualized text extraction from candidate records."""

from __future__ import annotations


def _skill_names(skills: list) -> list[str]:
    names: list[str] = []
    for skill in skills:
        if isinstance(skill, dict):
            name = skill.get("name", "").strip()
        else:
            name = str(skill).strip()
        if name:
            names.append(name)
    return names


def build_candidate_segments(record: dict) -> list[str]:
    """
    Extract role-contextualized segments from a candidate record.

    Adapted for the Redrob schema (career_history, profile, skills objects).
    """
    segments: list[str] = []

    for exp in record.get("career_history", record.get("experience", [])):
        title = exp.get("title", "").strip()
        company = exp.get("company", "").strip()
        description = exp.get("description", "").strip()
        if not description:
            continue
        context_prefix = f"{title} at {company}" if company else title
        segments.append(f"{context_prefix}: {description}")

    # skills = _skill_names(record.get("skills", []))
    # if skills:
    #     segments.append(f"technical skills include: {', '.join(skills)}")

    profile = record.get("profile", {})
    current_title = profile.get("current_title", record.get("current_title", "")).strip()
    if current_title:
        segments.append(f"current role: {current_title}")

    summary = profile.get("summary", "").strip()
    if summary:
        segments.append(f"professional summary: {summary}")

    # headline = profile.get("headline", "").strip()
    # if headline:
    #     segments.append(f"headline: {headline}")

    return segments


def build_candidate_passage(record: dict) -> str:
    """Join role-contextualized segments into a single passage for INSTRUCTOR encoding."""
    segments = build_candidate_segments(record)
    if not segments:
        return ""
    return " ".join(segments)


def truncate_passage(text: str, tokenizer, max_tokens: int = 480) -> str:
    """Truncate from the end to preserve leading (most relevant) content."""
    if not text.strip():
        return text
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
