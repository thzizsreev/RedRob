"""Extract fact-grounded career evidence for reasoning strings."""

from __future__ import annotations

import re

from ranker import jd_config as jd


def _norm(s: str) -> str:
    return s.lower().strip()


def best_career_phrase(candidate: dict) -> str | None:
    """Return a short career snippet that matches JD-positive lexicon."""
    history = candidate.get("career_history", [])
    summary = candidate.get("profile", {}).get("summary", "")
    best: tuple[int, str] | None = None

    for text in [summary] + [h.get("description", "") for h in history]:
        if not text or len(text) < 20:
            continue
        text_n = _norm(text)
        hits = sum(1 for term in jd.CAREER_POSITIVE if term in text_n)
        if hits == 0:
            continue
        snippet = text.strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        score = hits * 10 + min(len(snippet), 80)
        if best is None or score > best[0]:
            best = (score, snippet)

    return best[1] if best else None


def trusted_skills_with_duration(candidate: dict, limit: int = 3) -> list[str]:
    """Top trusted skills formatted as 'Name (Nmo)'."""
    from ranker.features import _skill_trust

    ranked: list[tuple[float, str]] = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if not name:
            continue
        trust = _skill_trust(skill)
        months = skill.get("duration_months", 0)
        ranked.append((trust, f"{name} ({months}mo)"))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in ranked[:limit]]
