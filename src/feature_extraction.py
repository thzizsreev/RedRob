"""Extract structured features from normalized candidate records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.normalizer import CandidateRecord
from src.utils import (
    clean_text,
    find_matching_terms,
    parse_date,
    proficiency_weight,
    title_matches,
)


@dataclass
class CandidateFeatures:
    candidate_id: str

    # Text aggregates
    career_text: str = ""
    skills_text: str = ""

    # Term hits
    tier_a_career_hits: list[str] = field(default_factory=list)
    tier_b_career_hits: list[str] = field(default_factory=list)
    tier_c_career_hits: list[str] = field(default_factory=list)
    tier_a_skill_hits: list[str] = field(default_factory=list)
    tier_b_skill_hits: list[str] = field(default_factory=list)
    tier_c_skill_hits: list[str] = field(default_factory=list)
    plain_language_career_hits: list[str] = field(default_factory=list)
    production_phrase_hits: list[str] = field(default_factory=list)
    hands_on_hits: list[str] = field(default_factory=list)
    buzzword_only_hits: list[str] = field(default_factory=list)

    # Booleans
    has_production_language: bool = False
    is_engineering_title: bool = False
    is_disqualifier_title: bool = False
    years_in_ideal_range: bool = False
    has_consulting_background: bool = False
    has_product_company_experience: bool = False
    has_career_descriptions: bool = False
    salary_inconsistent: bool = False

    # Numeric
    years_of_experience: float = 0.0
    tier_a_skill_count: int = 0
    tier_a_career_count: int = 0
    expert_skill_count: int = 0
    profile_completeness: float = 0.0
    avg_assessment_score: float = -1.0
    unrealistic_skill_durations: int = 0

    # Title info
    current_title: str = ""
    matched_engineering_titles: list[str] = field(default_factory=list)

    # For reasoning
    top_career_terms: list[str] = field(default_factory=list)
    top_skill_names: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


def _collect_career_text(candidate: CandidateRecord) -> str:
    parts = []
    for exp in candidate.career_history:
        if exp.title:
            parts.append(exp.title)
        if exp.description:
            parts.append(exp.description)
    return clean_text(" ".join(parts))


def _collect_titles_text(candidate: CandidateRecord) -> str:
    titles = [candidate.profile.current_title]
    titles.extend(exp.title for exp in candidate.career_history if exp.title)
    return clean_text(" ".join(titles))


def extract_features(candidate: CandidateRecord, jd_terms: dict) -> CandidateFeatures:
    tier_a = (
        jd_terms.get("tier_a_retrieval", [])
        + jd_terms.get("plain_language_production", [])
    )
    tier_b = jd_terms.get("tier_b_engineering", [])
    tier_c = jd_terms.get("tier_c_limited", [])
    plain = jd_terms.get("plain_language_production", [])
    production = jd_terms.get("production_phrases", [])
    hands_on = jd_terms.get("hands_on_phrases", [])
    buzzword_only = jd_terms.get("buzzword_only", [])
    engineering_titles = jd_terms.get("engineering_titles", [])
    disqualifier_titles = jd_terms.get("disqualifier_titles", [])
    consulting = jd_terms.get("consulting_indicators", [])
    product_sizes = jd_terms.get("product_company_sizes", [])

    career_text = _collect_career_text(candidate)
    skills_text = clean_text(
        " ".join(s.name for s in candidate.skills)
    )
    titles_text = _collect_titles_text(candidate)
    combined = candidate.combined_text

    years = candidate.profile.years_of_experience
    ideal_min = 5.0
    ideal_max = 9.0

    tier_a_career = find_matching_terms(career_text, tier_a)
    tier_b_career = find_matching_terms(career_text, tier_b)
    tier_c_career = find_matching_terms(career_text, tier_c)
    tier_a_skill = find_matching_terms(skills_text, tier_a)
    tier_b_skill = find_matching_terms(skills_text, tier_b)
    tier_c_skill = find_matching_terms(skills_text, tier_c)
    plain_career = find_matching_terms(career_text, plain)
    production_hits = find_matching_terms(career_text + " " + combined, production)
    hands_on_hits = find_matching_terms(career_text, hands_on)
    buzzword_hits = find_matching_terms(combined, buzzword_only)

    is_engineering = title_matches(titles_text, engineering_titles)
    is_disqualifier = title_matches(
        candidate.profile.current_title, disqualifier_titles
    )

    matched_eng = [
        t for t in engineering_titles
        if title_matches(titles_text, [t])
    ]

    has_consulting = any(
        contains_indicator(candidate.profile.current_industry, consulting)
        or contains_indicator(exp.industry, consulting)
        for exp in candidate.career_history
    ) or contains_indicator(candidate.profile.current_industry, consulting)

    has_product = (
        candidate.profile.current_company_size in product_sizes
        or any(exp.company_size in product_sizes for exp in candidate.career_history)
    )

    max_months = years * 12 if years > 0 else 0
    unrealistic = 0
    expert_count = 0
    for skill in candidate.skills:
        if skill.proficiency in ("expert", "advanced"):
            expert_count += 1
        if (
            max_months > 0
            and skill.duration_months > max_months + 12
            and skill.proficiency == "expert"
        ):
            unrealistic += 1

    signals = candidate.redrob_signals
    salary_bad = (
        signals.expected_salary_min > 0
        and signals.expected_salary_max > 0
        and signals.expected_salary_min > signals.expected_salary_max
    )

    assessments = list(signals.skill_assessment_scores.values())
    avg_assessment = (
        sum(assessments) / len(assessments) if assessments else -1.0
    )

    top_skills = sorted(
        candidate.skills,
        key=lambda s: (
            proficiency_weight(s.proficiency),
            s.endorsements,
            s.duration_months,
        ),
        reverse=True,
    )

    return CandidateFeatures(
        candidate_id=candidate.candidate_id,
        career_text=career_text,
        skills_text=skills_text,
        tier_a_career_hits=tier_a_career,
        tier_b_career_hits=tier_b_career,
        tier_c_career_hits=tier_c_career,
        tier_a_skill_hits=tier_a_skill,
        tier_b_skill_hits=tier_b_skill,
        tier_c_skill_hits=tier_c_skill,
        plain_language_career_hits=plain_career,
        production_phrase_hits=production_hits,
        hands_on_hits=hands_on_hits,
        buzzword_only_hits=buzzword_hits,
        has_production_language=len(production_hits) > 0,
        is_engineering_title=is_engineering,
        is_disqualifier_title=is_disqualifier,
        years_in_ideal_range=ideal_min <= years <= ideal_max,
        has_consulting_background=has_consulting,
        has_product_company_experience=has_product,
        has_career_descriptions=bool(career_text.strip()),
        salary_inconsistent=salary_bad,
        years_of_experience=years,
        tier_a_skill_count=len(tier_a_skill),
        tier_a_career_count=len(tier_a_career),
        expert_skill_count=expert_count,
        profile_completeness=signals.profile_completeness_score,
        avg_assessment_score=avg_assessment,
        unrealistic_skill_durations=unrealistic,
        current_title=candidate.profile.current_title,
        matched_engineering_titles=matched_eng,
        top_career_terms=(tier_a_career + plain_career)[:3],
        top_skill_names=[s.name for s in top_skills[:3]],
    )


def contains_indicator(text: str, indicators: list[str]) -> bool:
    text_lower = clean_text(text)
    return any(ind.lower() in text_lower for ind in indicators)


def days_since_active(last_active: str, reference: date | None = None) -> int:
    ref = reference or date.today()
    active = parse_date(last_active)
    if active is None:
        return 365
    return max(0, (ref - active).days)
