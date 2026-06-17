"""Normalize raw candidate dicts into typed records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils import clean_text, safe_float, safe_int


@dataclass
class SkillRecord:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int


@dataclass
class CareerRecord:
    company: str
    title: str
    description: str
    industry: str
    company_size: str
    duration_months: int
    is_current: bool


@dataclass
class ProfileRecord:
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


@dataclass
class RedrobSignals:
    profile_completeness_score: float
    last_active_date: str
    open_to_work_flag: bool
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    notice_period_days: int
    expected_salary_min: float
    expected_salary_max: float
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    verified_email: bool
    verified_phone: bool


@dataclass
class CandidateRecord:
    candidate_id: str
    profile: ProfileRecord
    career_history: list[CareerRecord]
    skills: list[SkillRecord]
    education: list[dict[str, Any]]
    redrob_signals: RedrobSignals
    combined_text: str = ""
    raw: dict = field(default_factory=dict, repr=False)


def _parse_skills(skills: list) -> list[SkillRecord]:
    result: list[SkillRecord] = []
    for skill in skills or []:
        if isinstance(skill, dict):
            name = str(skill.get("name", "")).strip()
            if not name:
                continue
            result.append(
                SkillRecord(
                    name=name,
                    proficiency=str(skill.get("proficiency", "intermediate")),
                    endorsements=safe_int(skill.get("endorsements")),
                    duration_months=safe_int(skill.get("duration_months")),
                )
            )
        elif isinstance(skill, str) and skill.strip():
            result.append(
                SkillRecord(
                    name=skill.strip(),
                    proficiency="intermediate",
                    endorsements=0,
                    duration_months=0,
                )
            )
    return result


def _parse_career(career: list) -> list[CareerRecord]:
    result: list[CareerRecord] = []
    for exp in career or []:
        if not isinstance(exp, dict):
            continue
        result.append(
            CareerRecord(
                company=str(exp.get("company", "")).strip(),
                title=str(exp.get("title", "")).strip(),
                description=str(exp.get("description", "")).strip(),
                industry=str(exp.get("industry", "")).strip(),
                company_size=str(exp.get("company_size", "")).strip(),
                duration_months=safe_int(exp.get("duration_months")),
                is_current=bool(exp.get("is_current")),
            )
        )
    return result


def _build_combined_text(
    profile: ProfileRecord,
    career_history: list[CareerRecord],
    skills: list[SkillRecord],
    education: list,
) -> str:
    parts: list[str] = []
    if profile.headline:
        parts.append(profile.headline)
    if profile.summary:
        parts.append(profile.summary)
    if profile.current_title:
        parts.append(profile.current_title)
    for exp in career_history:
        if exp.title:
            parts.append(exp.title)
        if exp.description:
            parts.append(exp.description)
    if skills:
        parts.append(", ".join(s.name for s in skills))
    for edu in education or []:
        if isinstance(edu, dict):
            inst = edu.get("institution", "")
            degree = edu.get("degree", "")
            field_of_study = edu.get("field_of_study", "")
            parts.append(f"{degree} {field_of_study} {inst}".strip())
    return clean_text(" ".join(p for p in parts if p))


def normalize_candidate(raw: dict) -> CandidateRecord:
    profile_data = raw.get("profile", {})
    signals_data = raw.get("redrob_signals", {})
    salary = signals_data.get("expected_salary_range_inr_lpa") or {}

    profile = ProfileRecord(
        headline=str(profile_data.get("headline", "")).strip(),
        summary=str(profile_data.get("summary", "")).strip(),
        location=str(profile_data.get("location", "")).strip(),
        country=str(profile_data.get("country", "")).strip(),
        years_of_experience=safe_float(profile_data.get("years_of_experience")),
        current_title=str(profile_data.get("current_title", "")).strip(),
        current_company=str(profile_data.get("current_company", "")).strip(),
        current_company_size=str(profile_data.get("current_company_size", "")).strip(),
        current_industry=str(profile_data.get("current_industry", "")).strip(),
    )

    career_history = _parse_career(raw.get("career_history", raw.get("experience", [])))
    skills = _parse_skills(raw.get("skills", []))
    education = list(raw.get("education", []) or [])

    assessment_raw = signals_data.get("skill_assessment_scores") or {}
    assessments = {
        str(k): safe_float(v) for k, v in assessment_raw.items() if k
    }

    signals = RedrobSignals(
        profile_completeness_score=safe_float(
            signals_data.get("profile_completeness_score")
        ),
        last_active_date=str(signals_data.get("last_active_date", "")),
        open_to_work_flag=bool(signals_data.get("open_to_work_flag")),
        recruiter_response_rate=safe_float(
            signals_data.get("recruiter_response_rate")
        ),
        avg_response_time_hours=safe_float(
            signals_data.get("avg_response_time_hours")
        ),
        skill_assessment_scores=assessments,
        notice_period_days=safe_int(signals_data.get("notice_period_days")),
        expected_salary_min=safe_float(salary.get("min")),
        expected_salary_max=safe_float(salary.get("max")),
        preferred_work_mode=str(
            signals_data.get("preferred_work_mode", "")
        ).strip(),
        willing_to_relocate=bool(signals_data.get("willing_to_relocate")),
        github_activity_score=safe_float(
            signals_data.get("github_activity_score"), -1.0
        ),
        saved_by_recruiters_30d=safe_int(
            signals_data.get("saved_by_recruiters_30d")
        ),
        interview_completion_rate=safe_float(
            signals_data.get("interview_completion_rate")
        ),
        verified_email=bool(signals_data.get("verified_email")),
        verified_phone=bool(signals_data.get("verified_phone")),
    )

    combined_text = _build_combined_text(profile, career_history, skills, education)

    return CandidateRecord(
        candidate_id=str(raw["candidate_id"]),
        profile=profile,
        career_history=career_history,
        skills=skills,
        education=education,
        redrob_signals=signals,
        combined_text=combined_text,
        raw=raw,
    )
