"""Assemble team_results-shaped candidate dicts for Stage 6 reasoning."""

from __future__ import annotations

from typing import Any

PIPELINE_SCORE_KEYS = (
    "stage3_rank",
    "stage4_rank",
    "cross_encoder_score",
    "fused_score",
    "q1_score",
    "q1_rank",
    "q2_score",
    "q2_rank",
    "skill_score",
    "skill_rank",
    "rrf_score",
    "q3_neg_sim",
)

STAGE5_SCORE_KEYS = (
    "final_score",
    "borda_primary",
    "borda_sum",
    "t1_std",
    "rank_ce",
    "rank_q1",
    "rank_q2",
    "rank_q1_amp",
    "rank_q2_amp",
    "sweet_bonus",
    "tier2_raw",
    "tier2_scaled",
    "t2_std",
    "target_t2_std",
    "avail_tier",
    "avail_unit",
    "tier3_scaled",
    "days_since_active",
    "location_unit",
    "workmode_unit",
    "notice_unit",
    "tier4_raw",
    "tier4_scaled",
    "in_top_100",
)

GATE_KEYS = (
    "total_years_exp",
    "exp_band",
    "in_sweet_spot",
    "title_family",
    "skill_kw_density",
    "title_ambiguous",
    "stale_profile",
    "low_responder",
    "not_open",
    "honeypot_anomaly_score",
    "product_company_count",
    "consulting_company_count",
    "product_company_fraction",
    "career_type",
    "research_fraction",
    "research_heavy",
    "has_any_production_role",
    "stale_coding",
    "currently_between_roles",
    "months_since_last_ic_role",
    "pre_llm_production_ml",
    "recent_ai_only",
    "llm_framework_only",
    "ml_experience_start_year",
    "avg_tenure_per_employer",
    "short_hop_count",
    "title_progression_jumps",
    "location_tier",
    "external_validation_score",
    "has_github",
    "notice_period_days",
    "cluster_id",
    "cluster_rank",
    "dist_to_centroid",
)

SIGNAL_KEYS = (
    "open_to_work_flag",
    "last_active_date",
    "applications_submitted_30d",
    "recruiter_response_rate",
    "avg_response_time_hours",
    "interview_completion_rate",
    "offer_acceptance_rate",
    "preferred_work_mode",
    "profile_completeness_score",
    "profile_views_received_30d",
    "saved_by_recruiters_30d",
    "github_activity_score",
    "notice_period_days",
    "expected_salary_range_inr_lpa",
    "willing_to_relocate",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
)


def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys if key in row}


def assemble_candidate_dict(
    stage5_row: dict[str, Any],
    jsonl_record: dict[str, Any],
    skill_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Map Stage 5 top row + JSONL profile into reasoning_builder input shape."""
    profile = jsonl_record.get("profile") or {}
    signals = jsonl_record.get("redrob_signals") or {}

    gates = _pick(stage5_row, GATE_KEYS)
    for key in GATE_KEYS:
        if gates.get(key) is None and key in stage5_row:
            gates[key] = stage5_row[key]

    behavioral = _pick(signals, SIGNAL_KEYS)
    for key in SIGNAL_KEYS:
        if behavioral.get(key) is None and key in stage5_row:
            behavioral[key] = stage5_row[key]

    retrieval = _pick(stage5_row, PIPELINE_SCORE_KEYS)
    if "cross_encoder_score" not in retrieval and "cross_encoder_score" in stage5_row:
        retrieval["cross_encoder_score"] = stage5_row["cross_encoder_score"]

    stage5_scoring = _pick(stage5_row, STAGE5_SCORE_KEYS)

    pipeline: dict[str, Any] = {
        "retrieval_scores": retrieval,
        "gates_and_career": gates,
        "behavioral_signals": behavioral,
        "stage5_scoring": stage5_scoring,
    }
    if skill_scores:
        pipeline["skill_assessment_scores"] = skill_scores

    return {
        "candidate_id": str(stage5_row["candidate_id"]),
        "profile": profile,
        "career_history": jsonl_record.get("career_history") or [],
        "skills": jsonl_record.get("skills") or [],
        "pipeline": pipeline,
    }
