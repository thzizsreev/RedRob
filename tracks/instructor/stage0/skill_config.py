"""Load stage0_skill configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Stage0SkillConfig:
    top_k_skills: int
    tier_rewards: dict[str, float]
    tier_keywords: dict[str, list[str]]


def load_stage0_skill_config(config_path: Path) -> Stage0SkillConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "stage0_skill" not in raw:
        raise ValueError(f"Missing 'stage0_skill' namespace in {config_path}")

    s0 = raw["stage0_skill"]
    tier_rewards = {str(k): float(v) for k, v in s0["tier_rewards"].items()}
    tier_keywords = {
        str(tier): [str(kw) for kw in keywords]
        for tier, keywords in s0["tier_keywords"].items()
    }
    return Stage0SkillConfig(
        top_k_skills=int(s0.get("top_k_skills", 15)),
        tier_rewards=tier_rewards,
        tier_keywords=tier_keywords,
    )
