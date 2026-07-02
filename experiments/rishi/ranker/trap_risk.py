"""Trap-Risk Modulator (TRM) — soft honeypot risk for TRACER scoring."""

from __future__ import annotations

from ranker import jd_config as jd
from ranker.honeypot import soft_trap_flags


def compute_trap_risk(
    candidate: dict,
    features: dict,
    semantic_score: float,
) -> tuple[float, list[str]]:
    """Return trap_risk in [0, 0.45] and contributing reasons."""
    risk = 0.0
    reasons: list[str] = []

    for flag in soft_trap_flags(candidate):
        if flag == "career_desc_reuse":
            risk += 0.15
            reasons.append("career_desc_reuse")
        elif flag == "keyword_stuffer":
            risk += 0.20
            reasons.append("keyword_stuffer")

    title = features.get("title_score", 0.0)
    skill = features.get("skill_score", 0.0)
    if title < jd.STUFFER_TITLE_THRESHOLD and skill > jd.STUFFER_SKILL_THRESHOLD:
        risk += 0.15
        reasons.append("title_skill_mismatch")

    if title <= 0.10 and semantic_score < 0.58:
        risk += 0.10
        reasons.append("weak_title_low_semantic")

    return min(0.45, risk), reasons
