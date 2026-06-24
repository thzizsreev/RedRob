"""Check F — research-only career hard remove + research composition features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tracks.instructor.stage2.checks._history import iter_roles_sorted, normalize_text
from tracks.instructor.stage2.checks.consulting import classify_employer
from tracks.instructor.stage2.config import ResearchConfig, Stage2Config

RoleType = Literal["research_role", "production_role", "unknown"]


@dataclass(frozen=True)
class ResearchResult:
    research_fraction: float
    research_heavy: bool
    has_any_production_role: bool
    remove: bool
    reason: str | None


def _title_matches_any(title: str, signals: list[str]) -> bool:
    if not title:
        return False
    return any(signal in title for signal in signals)


def _classify_role(
    role: dict,
    research_cfg: ResearchConfig,
    consulting_cfg,
) -> RoleType:
    title = normalize_text(role.get("title"))
    company = normalize_text(role.get("company"))
    employer_kind = classify_employer(company, consulting_cfg)

    has_production_title = _title_matches_any(title, research_cfg.production_title_signals)
    has_research_title = _title_matches_any(title, research_cfg.research_title_signals)
    has_academic_employer = _title_matches_any(company, research_cfg.academic_employer_signals)

    if employer_kind == "product" and has_production_title:
        return "production_role"

    if "research scientist" in title and employer_kind == "product":
        if has_production_title:
            return "production_role"

    if has_research_title or has_academic_employer:
        return "research_role"

    if has_production_title and employer_kind != "consulting":
        return "production_role"

    return "unknown"


def evaluate_research(record: dict, config: Stage2Config) -> ResearchResult:
    career_history = record.get("career_history") or []
    roles = iter_roles_sorted(career_history)
    research_cfg = config.research

    classified: list[RoleType] = []
    for role in roles:
        classified.append(_classify_role(role, research_cfg, config.consulting))

    classifiable = [r for r in classified if r != "unknown"]
    if len(classifiable) < research_cfg.min_roles_to_classify:
        return ResearchResult(
            research_fraction=0.0,
            research_heavy=False,
            has_any_production_role=False,
            remove=False,
            reason=None,
        )

    research_count = sum(1 for r in classifiable if r == "research_role")
    production_count = sum(1 for r in classifiable if r == "production_role")
    total = len(classifiable)
    fraction = research_count / total if total > 0 else 0.0
    has_production = production_count > 0

    remove = False
    reason: str | None = None
    if not has_production and research_count >= 1 and research_count == total:
        remove = True
        reason = "research_only_career"

    research_heavy = (
        not remove
        and has_production
        and fraction > research_cfg.research_heavy_threshold
    )

    return ResearchResult(
        research_fraction=fraction,
        research_heavy=research_heavy,
        has_any_production_role=has_production,
        remove=remove,
        reason=reason,
    )
