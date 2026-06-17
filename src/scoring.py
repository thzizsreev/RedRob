"""Sub-score computation for technical, career, behavioral, and logistics fit."""

from __future__ import annotations

from dataclasses import dataclass

from src.feature_extraction import CandidateFeatures, days_since_active
from src.normalizer import CandidateRecord
from src.utils import proficiency_weight


@dataclass
class ScoreBreakdown:
    technical: float = 0.0
    career: float = 0.0
    behavioral: float = 0.0
    logistics: float = 0.0
    risk_penalty: float = 0.0
    final_score: float = 0.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_technical(
    features: CandidateFeatures,
    candidate: CandidateRecord,
    jd_terms: dict,
    weights: dict,
) -> float:
    cfg = weights.get("technical", {})
    tier_a_w = cfg.get("tier_a_weight", 3.0)
    tier_b_w = cfg.get("tier_b_weight", 1.5)
    tier_c_w = cfg.get("tier_c_weight", 0.5)
    tier_c_cap = cfg.get("tier_c_cap_without_tier_a", 15.0)
    prod_mult = cfg.get("production_multiplier", 1.5)
    career_w = cfg.get("career_evidence_weight", 0.70)
    skills_w = cfg.get("skills_evidence_weight", 0.30)
    max_score = cfg.get("max_score", 100.0)

    def term_score(hits_a: list, hits_b: list, hits_c: list) -> float:
        score = len(hits_a) * tier_a_w + len(hits_b) * tier_b_w
        c_contrib = len(hits_c) * tier_c_w
        if not hits_a:
            c_contrib = min(c_contrib, tier_c_cap)
        return score

    career_raw = term_score(
        features.tier_a_career_hits,
        features.tier_b_career_hits,
        features.tier_c_career_hits,
    )
    if features.has_production_language:
        career_raw *= prod_mult
    career_raw += len(features.plain_language_career_hits) * cfg.get(
        "plain_language_weight", 2.0
    )

    skills_raw = 0.0
    tier_a_set = set(
        t.lower()
        for t in jd_terms.get("tier_a_retrieval", [])
        + jd_terms.get("plain_language_production", [])
    )
    tier_b_set = set(t.lower() for t in jd_terms.get("tier_b_engineering", []))
    tier_c_set = set(t.lower() for t in jd_terms.get("tier_c_limited", []))

    for skill in candidate.skills:
        name_lower = skill.name.lower()
        pw = proficiency_weight(skill.proficiency)
        if any(term in name_lower or name_lower in term for term in tier_a_set):
            skills_raw += tier_a_w * pw
        elif any(term in name_lower or name_lower in term for term in tier_b_set):
            skills_raw += tier_b_w * pw
        elif any(term in name_lower or name_lower in term for term in tier_c_set):
            skills_raw += tier_c_w * pw

    if features.tier_a_skill_count >= 3 and features.tier_a_career_count == 0:
        skills_raw *= 0.4

    if features.buzzword_only_hits and not features.has_production_language:
        career_raw *= 0.5
        skills_raw *= 0.5

    combined = career_w * career_raw + skills_w * skills_raw
    return _clamp(combined, 0.0, max_score)


def score_career(
    features: CandidateFeatures,
    candidate: CandidateRecord,
    jd_terms: dict,
    weights: dict,
) -> float:
    cfg = weights.get("career", {})
    years = features.years_of_experience
    ideal_min = cfg.get("ideal_years_min", 5.0)
    ideal_max = cfg.get("ideal_years_max", 9.0)

    score = 0.0

    if ideal_min <= years <= ideal_max:
        score += cfg.get("years_peak_score", 30.0)
    elif years < ideal_min:
        score += cfg.get("years_peak_score", 30.0) * (years / ideal_min)
    else:
        over = years - ideal_max
        score += max(0.0, cfg.get("years_peak_score", 30.0) - over * 5.0)

    if features.is_engineering_title:
        score += cfg.get("title_match_score", 25.0)
    if features.is_disqualifier_title:
        score -= cfg.get("non_technical_title_penalty", 40.0)

    if features.hands_on_hits:
        score += min(
            cfg.get("hands_on_score", 20.0),
            len(features.hands_on_hits) * 5.0,
        )

    if features.has_product_company_experience:
        score += cfg.get("product_company_bonus", 15.0)
    if features.has_consulting_background and not features.has_product_company_experience:
        score -= cfg.get("consulting_penalty", 15.0)

    return _clamp(score, 0.0, cfg.get("max_score", 100.0))


def score_behavioral(
    candidate: CandidateRecord,
    weights: dict,
    technical_score: float,
) -> float:
    cfg = weights.get("behavioral", {})
    rescue_cfg = weights.get("behavioral_rescue", {})
    min_technical = rescue_cfg.get("min_technical", 20.0)

    signals = candidate.redrob_signals
    score = 0.0

    if signals.open_to_work_flag:
        score += cfg.get("open_to_work", 15.0)

    days_inactive = days_since_active(signals.last_active_date)
    if days_inactive <= 30:
        score += cfg.get("activity_recency", 20.0)
    elif days_inactive <= 90:
        score += cfg.get("activity_recency", 20.0) * 0.6
    elif days_inactive <= 180:
        score += cfg.get("activity_recency", 20.0) * 0.3

    score += signals.recruiter_response_rate * cfg.get("response_rate", 20.0)

    if signals.avg_response_time_hours <= 24:
        score += cfg.get("response_time", 15.0)
    elif signals.avg_response_time_hours <= 72:
        score += cfg.get("response_time", 15.0) * 0.6
    elif signals.avg_response_time_hours <= 168:
        score += cfg.get("response_time", 15.0) * 0.3

    score += signals.interview_completion_rate * cfg.get(
        "interview_completion", 10.0
    )

    if signals.github_activity_score >= 0:
        score += (signals.github_activity_score / 100.0) * cfg.get(
            "github_activity", 10.0
        )

    if signals.saved_by_recruiters_30d > 0:
        score += min(
            cfg.get("recruiter_saves", 5.0),
            signals.saved_by_recruiters_30d,
        )

    if signals.verified_email and signals.verified_phone:
        score += cfg.get("verification", 5.0)
    elif signals.verified_email or signals.verified_phone:
        score += cfg.get("verification", 5.0) * 0.5

    score = _clamp(score, 0.0, cfg.get("max_score", 100.0))

    if technical_score < min_technical:
        score *= technical_score / min_technical if min_technical > 0 else 0.0

    return score


def score_logistics(candidate: CandidateRecord, weights: dict) -> float:
    cfg = weights.get("logistics", {})
    signals = candidate.redrob_signals
    score = 0.0

    notice = signals.notice_period_days
    if notice <= 30:
        score += cfg.get("notice_period", 40.0)
    elif notice <= 60:
        score += cfg.get("notice_period", 40.0) * 0.75
    elif notice <= 90:
        score += cfg.get("notice_period", 40.0) * 0.5
    elif notice < 120:
        score += cfg.get("notice_period", 40.0) * 0.25

    if signals.willing_to_relocate:
        score += cfg.get("relocation", 30.0)

    mode = signals.preferred_work_mode.lower()
    if mode in ("hybrid", "flexible", "remote"):
        score += cfg.get("work_mode", 30.0)
    elif mode == "onsite":
        score += cfg.get("work_mode", 30.0) * 0.5

    return _clamp(score, 0.0, cfg.get("max_score", 100.0))


def compute_final_score(
    features: CandidateFeatures,
    candidate: CandidateRecord,
    jd_terms: dict,
    weights: dict,
    risk_penalty: float,
) -> ScoreBreakdown:
    components = weights.get("components", {})
    technical = score_technical(features, candidate, jd_terms, weights)
    career = score_career(features, candidate, jd_terms, weights)
    behavioral = score_behavioral(candidate, weights, technical)
    logistics = score_logistics(candidate, weights)

    w_t = components.get("technical", 0.45)
    w_c = components.get("career", 0.25)
    w_b = components.get("behavioral", 0.20)
    w_l = components.get("logistics", 0.10)

    final = (
        w_t * technical
        + w_c * career
        + w_b * behavioral
        + w_l * logistics
        - risk_penalty
    )

    return ScoreBreakdown(
        technical=technical,
        career=career,
        behavioral=behavioral,
        logistics=logistics,
        risk_penalty=risk_penalty,
        final_score=final,
    )
