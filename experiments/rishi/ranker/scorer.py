"""TRACER scorer — Trap-aware Retrieval And Career Evidence Ranker."""

from __future__ import annotations

from ranker import jd_config as jd
from ranker.behavioral import behavioral_multiplier
from ranker.features import extract_features
from ranker.honeypot import is_honeypot, soft_trap_flags
from ranker.trap_risk import compute_trap_risk


def score_candidate(
    candidate: dict,
    semantic_score: float | None = None,
) -> tuple[float, dict]:
    """TRACER score: fused semantic + structured features × (1 − trap_risk) × behavioral."""
    honeypot, hp_flags = is_honeypot(candidate)
    features = extract_features(candidate)
    signals = candidate.get("redrob_signals", {})
    soft_flags = soft_trap_flags(candidate)

    sem = semantic_score if semantic_score is not None else 0.5

    if honeypot:
        return 0.0, {
            "features": features,
            "semantic_score": sem,
            "honeypot": True,
            "honeypot_flags": hp_flags,
            "soft_trap_flags": soft_flags,
            "trap_risk": 1.0,
            "trap_risk_reasons": hp_flags,
            "base_score": 0.0,
            "behavioral_mult": 0.0,
        }

    w = jd.WEIGHTS
    base = (
        w["semantic"] * sem
        + w["title"] * features["title_score"]
        + w["career"] * features["career_score"]
        + w["skill"] * features["skill_score"]
        + w["experience"] * features["experience_score"]
        + w["location"] * features["location_score"]
        + w["assessment"] * features["assessment_score"]
        - features["anti_pattern_penalty"]
    )

    if (
        features["title_score"] < jd.STUFFER_TITLE_THRESHOLD
        and features["skill_score"] > jd.STUFFER_SKILL_THRESHOLD
        and sem < jd.STUFFER_SEMANTIC_FLOOR
    ):
        base *= 0.35

    if "keyword_stuffer" in soft_flags and sem < 0.58:
        base *= 0.40

    if features["title_score"] <= 0.10 and sem < 0.62:
        base *= 0.25

    base = max(0.0, min(1.0, base))
    trap_risk, trap_reasons = compute_trap_risk(candidate, features, sem)
    bmult = behavioral_multiplier(signals)
    if base >= 0.75 and bmult < 0.50:
        bmult = 0.50

    final = base * (1.0 - trap_risk) * bmult

    return final, {
        "features": features,
        "semantic_score": sem,
        "honeypot": False,
        "honeypot_flags": hp_flags,
        "soft_trap_flags": soft_flags,
        "trap_risk": trap_risk,
        "trap_risk_reasons": trap_reasons,
        "base_score": base,
        "behavioral_mult": bmult,
    }
