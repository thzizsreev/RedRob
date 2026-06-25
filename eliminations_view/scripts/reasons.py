"""Reason code labels, categories, and human-readable summaries."""

from __future__ import annotations

from typing import Any

REASON_LABELS: dict[str, str] = {
    "honeypot_future_start": "Role start date in the future",
    "honeypot_duration_overshoot": "Role duration extends past today",
    "honeypot_role_overlap": "Overlapping role timelines",
    "honeypot_timeline_sum": "Total role years exceed claimed experience",
    "honeypot_graduation_vs_exp": "Experience exceeds time since graduation",
    "honeypot_expert_zero": "Expert skills with zero duration",
    "honeypot_skill_years_impossible": "Skill years exceed total experience",
    "honeypot_skill_years_ceiling": "Skill years exceed plausible ceiling",
    "honeypot_inverted_dates": "Role end before start",
    "honeypot_future_end_date": "Role end date in the future",
    "exp_out_of_band": "Experience outside required band",
    "keyword_stuffer": "Keyword stuffer title",
    "non_eng_title": "Non-engineering title",
    "consulting_only_career": "Consulting-only career",
    "research_only_career": "Research-only career",
    "shallow_recent_ai_only": "Shallow recent-AI-only profile",
    "not_in_retrieval_union": "Not in retrieval union (L1/L2/L4)",
    "below_fused_cut": "Below fused-score cutoff",
    "below_rerank_cutoff": "Below rerank top-N cutoff",
    "below_final_cutoff": "Below final submission top-N cutoff",
}

RULE_LABELS: dict[str, str] = {
    "future_start": "Future start",
    "duration_overshoot": "Duration overshoot",
    "role_overlap": "Role overlap",
    "timeline_sum": "Timeline sum",
    "graduation_vs_exp": "Grad vs exp",
    "expert_zero": "Expert zero dur.",
    "skill_years_impossible": "Skill years",
    "skill_years_ceiling": "Skill ceiling",
    "inverted_dates": "Inverted dates",
    "future_end_date": "Future end",
}

# Short labels for card badges (2–3 words).
BADGE_LABELS: dict[str, str] = {
    "honeypot_future_start": "Future start",
    "honeypot_duration_overshoot": "Duration overshoot",
    "honeypot_role_overlap": "Role overlap",
    "honeypot_timeline_sum": "Timeline sum",
    "honeypot_graduation_vs_exp": "Grad vs exp",
    "honeypot_expert_zero": "Expert zero dur.",
    "honeypot_skill_years_impossible": "Skill years",
    "honeypot_skill_years_ceiling": "Skill ceiling",
    "honeypot_inverted_dates": "Inverted dates",
    "honeypot_future_end_date": "Future end",
    "exp_out_of_band": "Exp band",
    "keyword_stuffer": "Keyword stuffer",
    "non_eng_title": "Non-eng title",
    "consulting_only_career": "Consulting only",
    "research_only_career": "Research only",
    "shallow_recent_ai_only": "Shallow AI",
    "not_in_retrieval_union": "No retrieval",
    "below_fused_cut": "Low fused score",
    "below_rerank_cutoff": "Below rerank N",
    "below_final_cutoff": "Below top 100",
}


def category_for_reason(reason_code: str) -> str:
    if reason_code.startswith("honeypot_"):
        return "honeypot"
    if reason_code == "exp_out_of_band":
        return "experience"
    if reason_code in ("keyword_stuffer", "non_eng_title"):
        return "title"
    if reason_code == "consulting_only_career":
        return "consulting"
    if reason_code == "research_only_career":
        return "research"
    if reason_code == "shallow_recent_ai_only":
        return "shallow_ai"
    if reason_code in ("not_in_retrieval_union", "below_fused_cut"):
        return "retrieval"
    if reason_code == "below_rerank_cutoff":
        return "rerank"
    if reason_code == "below_final_cutoff":
        return "final_score"
    return "gate"


def badge_for_reason(reason_code: str, rules: list[str] | None = None) -> str:
    if reason_code in BADGE_LABELS:
        return BADGE_LABELS[reason_code]
    if reason_code.startswith("honeypot_") and rules:
        return RULE_LABELS.get(rules[0], rules[0].replace("_", " "))
    if reason_code.startswith("honeypot_"):
        return RULE_LABELS.get(
            reason_code.removeprefix("honeypot_"),
            reason_code.removeprefix("honeypot_").replace("_", " "),
        )
    return label_for_reason(reason_code)


def label_for_reason(reason_code: str) -> str:
    return REASON_LABELS.get(reason_code, reason_code.replace("_", " ").title())


def parse_honeypot_rules(reason_code: str, honeypot_rules: list[str] | None) -> list[str]:
    if honeypot_rules:
        return honeypot_rules
    if reason_code.startswith("honeypot_"):
        return [reason_code.removeprefix("honeypot_")]
    return []


def build_summary(
    reason_code: str,
    details: dict[str, Any] | None = None,
    pipeline: dict[str, Any] | None = None,
) -> str:
    details = details or {}
    pipeline = pipeline or {}

    if reason_code == "honeypot_skill_years_impossible":
        d = details.get("skill_years_impossible") or details
        skill = d.get("skill", "")
        max_y = d.get("max_skill_years", "")
        yoe = d.get("total_years_exp", "")
        return f"Max skill {max_y}y ({skill}) > claimed YOE {yoe}y"

    if reason_code == "honeypot_skill_years_ceiling":
        d = details.get("skill_years_ceiling") or details
        skill = d.get("skill", "")
        skill_y = d.get("skill_years", "")
        if d.get("violation") == "absolute":
            return f"Skill {skill_y}y ({skill}) exceeds absolute max 30y"
        ceiling = d.get("ceiling_years", "")
        yoe = d.get("total_years_exp", "")
        return f"Skill {skill_y}y ({skill}) exceeds ceiling {ceiling}y at {yoe} YOE"

    if reason_code == "honeypot_timeline_sum":
        d = details.get("timeline_sum") or details
        return (
            f"Role total {d.get('total_years_from_roles')}y > "
            f"claimed {d.get('claimed_years')}y "
            f"(tolerance {d.get('tolerance_years')}y)"
        )

    if reason_code == "honeypot_graduation_vs_exp":
        d = details.get("graduation_vs_exp") or details
        return (
            f"Claimed {d.get('claimed_years')}y > max possible "
            f"{d.get('max_possible_experience')}y after grad {d.get('latest_graduation_year')}"
        )

    if reason_code == "honeypot_expert_zero":
        d = details.get("expert_zero") or details
        return f"{d.get('count', 0)} expert skill(s) with zero duration"

    if reason_code == "not_in_retrieval_union":
        return "Did not appear in any retrieval list (dense Q1/Q2 or BM25 Q4)"

    if reason_code == "below_fused_cut":
        fused = pipeline.get("fused_score")
        threshold = pipeline.get("threshold")
        rank = pipeline.get("stage3_rank")
        parts = []
        if fused is not None:
            parts.append(f"fused={fused}")
        if threshold is not None:
            parts.append(f"threshold={threshold}")
        if rank is not None:
            parts.append(f"rank={rank}")
        return "Below fused-score cutoff" + (f" ({', '.join(parts)})" if parts else "")

    if reason_code == "below_rerank_cutoff":
        ce = pipeline.get("cross_encoder_score")
        s3 = pipeline.get("stage3_rank")
        keep_n = pipeline.get("keep_n")
        parts = []
        if ce is not None:
            parts.append(f"CE={ce}")
        if s3 is not None:
            parts.append(f"stage3_rank={s3}")
        if keep_n is not None:
            parts.append(f"keep_n={keep_n}")
        return "Below rerank top-N" + (f" ({', '.join(parts)})" if parts else "")

    if reason_code == "below_final_cutoff":
        final = pipeline.get("final_score")
        s4 = pipeline.get("stage4_rank")
        top_n = pipeline.get("top_n")
        parts = []
        if final is not None:
            parts.append(f"final={final}")
        if s4 is not None:
            parts.append(f"stage4_rank={s4}")
        if top_n is not None:
            parts.append(f"top_n={top_n}")
        return "Below final top-N" + (f" ({', '.join(parts)})" if parts else "")

    return label_for_reason(reason_code)


def build_elimination_meta(
    reason_code: str,
    *,
    rules: list[str] | None = None,
    details: dict[str, Any] | None = None,
    pipeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = rules or []
    return {
        "reason_code": reason_code,
        "reason_label": label_for_reason(reason_code),
        "category": category_for_reason(reason_code),
        "badge_label": badge_for_reason(reason_code, rules),
        "rules": rules,
        "details": details or {},
        "summary": build_summary(reason_code, details, pipeline),
    }
