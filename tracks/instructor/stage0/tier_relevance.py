"""Stub tier-based skill relevance lookup (replaced later by Blocks A–D)."""

from __future__ import annotations

from tracks.instructor.stage0.skill_config import Stage0SkillConfig


def normalize_skill_name(name: str) -> str:
    return name.strip().lower()


def build_tier_lookup(config: Stage0SkillConfig) -> dict[str, float]:
    """Map normalized skill name substring matches to tier reward."""
    lookup: dict[str, float] = {}
    for tier in ("T1", "T2", "T3", "T4", "T5"):
        reward = config.tier_rewards.get(tier, 0.0)
        for keyword in config.tier_keywords.get(tier, []):
            lookup[keyword.lower()] = reward
    return lookup


def relevance_score(skill_name: str, tier_lookup: dict[str, float]) -> float:
    """Return highest matching tier reward for a skill name, or 0.0 if unknown."""
    name = normalize_skill_name(skill_name)
    if not name:
        return 0.0

    best = 0.0
    for keyword, reward in tier_lookup.items():
        if keyword in name and reward > best:
            best = reward
    return best
