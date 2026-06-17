"""Detect risky profiles and compute penalty scores."""

from __future__ import annotations

from src.feature_extraction import CandidateFeatures
from src.normalizer import CandidateRecord


def compute_risk_penalty(
    features: CandidateFeatures,
    candidate: CandidateRecord,
    weights: dict,
) -> tuple[float, list[str]]:
    cfg = weights.get("risk", {})
    max_penalty = cfg.get("max_penalty", 40.0)
    penalty = 0.0
    flags: list[str] = []

    if features.salary_inconsistent:
        penalty += cfg.get("salary_inconsistency", 15.0)
        flags.append("salary range inconsistency")

    if (
        features.tier_a_skill_count >= 8
        and features.tier_a_career_count == 0
        and not features.is_engineering_title
    ):
        penalty += cfg.get("keyword_stuffing", 20.0)
        flags.append("keyword stuffing without career evidence")

    if features.is_disqualifier_title and features.expert_skill_count >= 3:
        penalty += cfg.get("title_skill_mismatch", 25.0)
        flags.append("non-technical title with expert AI skills")

    if (
        features.profile_completeness < 40
        and features.tier_a_skill_count >= 5
        and features.tier_a_career_count == 0
    ):
        penalty += cfg.get("low_completeness_buzzwords", 10.0)
        flags.append("low profile completeness with heavy buzzwords")

    if features.unrealistic_skill_durations > 0:
        penalty += cfg.get("unrealistic_skill_duration", 10.0)
        flags.append("unrealistic skill duration vs experience")

    if (
        features.avg_assessment_score >= 0
        and features.avg_assessment_score < 30
        and features.tier_a_skill_count >= 3
    ):
        penalty += cfg.get("low_assessments", 5.0)
        flags.append("low skill assessment scores")

    if _is_honeypot(features, candidate):
        penalty += cfg.get("honeypot_heuristic", 30.0)
        flags.append("honeypot-like profile pattern")

    penalty = min(penalty, max_penalty)
    return penalty, flags


def _is_honeypot(features: CandidateFeatures, candidate: CandidateRecord) -> bool:
    if not features.has_career_descriptions and features.tier_a_skill_count >= 5:
        return True

    if features.years_of_experience > 25 and features.tier_a_skill_count >= 10:
        return True

    if (
        features.is_disqualifier_title
        and features.tier_a_skill_count >= 6
        and features.tier_a_career_count == 0
    ):
        return True

    empty_career = all(
        not exp.description.strip() for exp in candidate.career_history
    )
    if empty_career and features.expert_skill_count >= 5:
        return True

    return False
