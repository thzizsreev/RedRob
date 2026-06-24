"""Check H — shallow recent AI hard remove with pre-LLM escape hatch."""

from __future__ import annotations

from dataclasses import dataclass

from tracks.instructor.stage2.checks._history import (
    iter_roles_sorted,
    normalize_text,
    subtract_months,
)
from tracks.instructor.stage2.config import Stage2Config
from tracks.instructor.stage2.honeypot_rules import _parse_date


@dataclass(frozen=True)
class ShallowAiResult:
    pre_llm_production_ml: bool
    recent_ai_only: bool
    llm_framework_only: bool
    ml_experience_start_year: int | None
    remove: bool
    reason: str | None


def _skill_names(record: dict) -> list[str]:
    skills = record.get("skills") or []
    return [normalize_text(s.get("name")) for s in skills if s.get("name")]


def _is_ml_title(title: str, config: Stage2Config) -> bool:
    normalized = normalize_text(title)
    if not normalized:
        return False
    signals = config.shallow_ai.ml_title_signals
    if any(signal in normalized for signal in signals):
        return True
    if "ai" in normalized.split():
        return True
    return False


def _earliest_ml_role_date(record: dict, config: Stage2Config):
    earliest = None
    for role in iter_roles_sorted(record.get("career_history") or []):
        title = str(role.get("title", ""))
        if not _is_ml_title(title, config):
            continue
        start = _parse_date(role.get("start_date"))
        if start is not None and (earliest is None or start < earliest):
            earliest = start
    return earliest


def _count_skill_matches(skill_names: list[str], signals: list[str]) -> int:
    count = 0
    for skill in skill_names:
        if any(signal in skill for signal in signals):
            count += 1
    return count


def evaluate_shallow_ai(record: dict, config: Stage2Config) -> ShallowAiResult:
    cfg = config.shallow_ai
    skill_names = _skill_names(record)
    earliest_ml = _earliest_ml_role_date(record, config)

    has_ml_skill = _count_skill_matches(skill_names, cfg.pre_llm_skill_signals) > 0
    has_ml_title_history = earliest_ml is not None

    if not has_ml_title_history and not _has_any_ml_skill_signal(skill_names, config):
        return ShallowAiResult(
            pre_llm_production_ml=False,
            recent_ai_only=False,
            llm_framework_only=False,
            ml_experience_start_year=None,
            remove=False,
            reason=None,
        )

    ml_start_year = earliest_ml.year if earliest_ml else None
    recent_cutoff = subtract_months(config.current_date, cfg.recent_ai_window_months)

    recent_ai_only = False
    if earliest_ml is not None:
        recent_ai_only = earliest_ml >= recent_cutoff
    elif has_ml_skill:
        recent_ai_only = True

    pre_llm_skill = _count_skill_matches(skill_names, cfg.pre_llm_skill_signals) > 0
    pre_llm_role = False
    if earliest_ml is not None and earliest_ml.year < cfg.llm_era_start_year:
        pre_llm_role = True
    pre_llm_production_ml = pre_llm_skill or pre_llm_role

    llm_count = _count_skill_matches(skill_names, cfg.llm_framework_signals)
    llm_framework_only = (
        llm_count >= cfg.min_llm_framework_skills and not pre_llm_skill
    )

    remove = False
    reason: str | None = None
    if (
        recent_ai_only
        and not pre_llm_production_ml
        and llm_count >= cfg.min_llm_framework_skills
        and not pre_llm_skill
    ):
        remove = True
        reason = "shallow_recent_ai_only"

    return ShallowAiResult(
        pre_llm_production_ml=pre_llm_production_ml,
        recent_ai_only=recent_ai_only,
        llm_framework_only=llm_framework_only,
        ml_experience_start_year=ml_start_year,
        remove=remove,
        reason=reason,
    )


def _has_any_ml_skill_signal(skill_names: list[str], config: Stage2Config) -> bool:
    cfg = config.shallow_ai
    if _count_skill_matches(skill_names, cfg.llm_framework_signals) > 0:
        return True
    if _count_skill_matches(skill_names, cfg.pre_llm_skill_signals) > 0:
        return True
    jd_ml = [kw for kw in config.jd_keywords if kw in ("rag", "embeddings", "sentence-transformers")]
    return _count_skill_matches(skill_names, jd_ml) > 0
