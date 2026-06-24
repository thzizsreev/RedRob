"""Check B — title family + keyword-stuffer gate."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tracks.instructor.stage2.config import Stage2Config

_FAMILY_PRIORITY = ("core_eng", "adjacent_eng", "ambiguous", "non_eng")


@dataclass(frozen=True)
class TitleResult:
    title_family: str
    skill_kw_density: float
    title_ambiguous: bool
    remove: bool
    reason: str | None


def _normalize_title(title: str) -> str:
    lowered = title.lower().strip()
    return re.sub(r"[^a-z0-9\s/+-]", " ", lowered)


def classify_title_family(title: str, config: Stage2Config) -> str:
    normalized = _normalize_title(title)
    if not normalized:
        return "ambiguous"

    best_family = "ambiguous"
    best_len = 0
    for family in _FAMILY_PRIORITY:
        phrases = config.title_families.get(family, [])
        for phrase in sorted(phrases, key=len, reverse=True):
            if phrase in normalized and len(phrase) > best_len:
                best_family = family
                best_len = len(phrase)
    return best_family


def compute_skill_kw_density(record: dict, config: Stage2Config) -> float:
    skills = record.get("skills") or []
    if not skills:
        return 0.0

    keywords = config.jd_keywords
    matches = 0
    for skill in skills:
        name = str(skill.get("name", "")).lower()
        if any(kw in name for kw in keywords):
            matches += 1
    return matches / max(1, len(skills))


def evaluate_title(record: dict, config: Stage2Config) -> TitleResult:
    profile = record.get("profile") or {}
    title = str(profile.get("current_title", ""))
    family = classify_title_family(title, config)
    density = compute_skill_kw_density(record, config)
    title_ambiguous = family == "ambiguous"

    if family in ("core_eng", "adjacent_eng"):
        return TitleResult(
            title_family=family,
            skill_kw_density=density,
            title_ambiguous=title_ambiguous,
            remove=False,
            reason=None,
        )

    if family == "ambiguous":
        return TitleResult(
            title_family=family,
            skill_kw_density=density,
            title_ambiguous=True,
            remove=False,
            reason=None,
        )

    if density >= config.stuffer_density:
        return TitleResult(
            title_family=family,
            skill_kw_density=density,
            title_ambiguous=False,
            remove=True,
            reason="keyword_stuffer",
        )

    return TitleResult(
        title_family=family,
        skill_kw_density=density,
        title_ambiguous=False,
        remove=True,
        reason="non_eng_title",
    )
