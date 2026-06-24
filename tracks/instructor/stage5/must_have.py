"""Layer 2 must-have floor computation."""

from __future__ import annotations

from tracks.instructor.stage5.config import Stage5Config


def _skill_names(skills: list[dict] | None) -> list[str]:
    if not skills:
        return []
    return [str(s.get("name", "")).lower() for s in skills if s.get("name")]


def _category_hit(skill_name: str, keywords: list[str]) -> bool:
    return any(kw in skill_name for kw in keywords)


def skill_maps_to_must_have_category(skill_name: str, config: Stage5Config) -> str | None:
    for category, keywords in config.must_have_keywords.items():
        if _category_hit(skill_name, keywords):
            return category
    return None


def keyword_ratio(skills: list[dict] | None, config: Stage5Config) -> float:
    names = _skill_names(skills)
    if not names:
        return 0.0
    categories = config.must_have_keywords
    covered = 0
    for keywords in categories.values():
        if any(_category_hit(name, keywords) for name in names):
            covered += 1
    return covered / max(len(categories), 1)


def assessment_coverage(
    assessments: dict | None,
    skills: list[dict] | None,
    config: Stage5Config,
    semantic_cov: float,
) -> float:
    if not assessments:
        return semantic_cov

    relevant_scores: list[float] = []
    for skill_name, score in assessments.items():
        if skill_maps_to_must_have_category(str(skill_name).lower(), config) is None:
            continue
        try:
            relevant_scores.append(float(score))
        except (TypeError, ValueError):
            continue

    if not relevant_scores:
        skill_names = _skill_names(skills)
        has_must_have_skill = any(
            skill_maps_to_must_have_category(name, config) is not None for name in skill_names
        )
        if has_must_have_skill:
            return semantic_cov
        return semantic_cov

    return sum(relevant_scores) / (len(relevant_scores) * 100.0)


def must_have_floor_multiplier(
    keyword_cov: float,
    semantic_cov: float,
    assessment_cov: float,
    floor_min: float,
) -> tuple[float, float]:
    combined = max(keyword_cov, semantic_cov, assessment_cov)
    multiplier = floor_min + (1.0 - floor_min) * combined
    return combined, multiplier
