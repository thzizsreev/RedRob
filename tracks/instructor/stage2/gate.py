"""Stage 2 hard tabular gate — single-pass orchestrator."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import polars as pl

from tracks.instructor.stage2.checks.availability import evaluate_availability
from tracks.instructor.stage2.checks.career_shape import evaluate_career_shape
from tracks.instructor.stage2.checks.coding_recency import evaluate_coding_recency
from tracks.instructor.stage2.checks.consulting import evaluate_consulting
from tracks.instructor.stage2.checks.experience import evaluate_experience
from tracks.instructor.stage2.checks.logistics import evaluate_logistics
from tracks.instructor.stage2.checks.research import evaluate_research
from tracks.instructor.stage2.checks.shallow_ai import evaluate_shallow_ai
from tracks.instructor.stage2.checks.skills import evaluate_skill_honeypot
from tracks.instructor.stage2.checks.title import evaluate_title
from tracks.instructor.stage2.checks.validation import evaluate_external_validation
from tracks.instructor.stage2.config import Stage2Config, load_stage2_config
from tracks.instructor.stage2.honeypot_rules import evaluate_timeline_honeypot
from tracks.instructor.stage2.io import (
    compute_dist_to_centroid,
    iter_candidates_by_ids,
    load_stage1_filter,
    write_stage2_outputs,
)


@dataclass(frozen=True)
class Stage2Result:
    input_count: int
    survivor_count: int
    removal_counts: dict[str, int]
    elapsed_seconds: float
    output_dir: Path


def _evaluate_honeypot(record: dict, config: Stage2Config) -> tuple[bool, list[str], dict]:
    timeline = evaluate_timeline_honeypot(record, config)
    skills = evaluate_skill_honeypot(record, config)

    rules = timeline.rules_fired + skills.rules_fired
    details = {**timeline.details, **skills.details}
    exclude = timeline.exclude or skills.exclude
    return exclude, rules, details


def _honeypot_reason_code(rules: list[str]) -> str:
    if not rules:
        return "honeypot"
    return f"honeypot_{rules[0]}"


def _validate_counts(
    input_count: int,
    survivor_count: int,
    config: Stage2Config,
) -> None:
    expected = config.expected_input_count
    low = expected * 0.8
    high = expected * 1.2
    if input_count < low or input_count > high:
        warnings.warn(
            f"Stage 2 input count ({input_count}) deviates >20% from "
            f"expected ~{expected}. Upstream Stage 1 may have changed.",
            stacklevel=2,
        )

    if survivor_count < config.expected_survivor_min:
        warnings.warn(
            f"Stage 2 survivor count ({survivor_count}) is below "
            f"expected minimum ({config.expected_survivor_min}). Thresholds may be too aggressive.",
            stacklevel=2,
        )
    if survivor_count > config.expected_survivor_max:
        warnings.warn(
            f"Stage 2 survivor count ({survivor_count}) is above "
            f"expected maximum ({config.expected_survivor_max}). Thresholds may be too permissive.",
            stacklevel=2,
        )


def run(
    *,
    stage1_path: Path,
    artifacts_path: Path,
    candidates_path: Path,
    output_dir: Path,
    config_path: Path,
) -> Stage2Result:
    start = perf_counter()
    config = load_stage2_config(config_path)

    filtered_ids, metadata = load_stage1_filter(stage1_path)
    id_set = set(filtered_ids)
    input_count = len(filtered_ids)

    dist_map = compute_dist_to_centroid(filtered_ids, artifacts_path, stage1_path)

    survivors: list[dict] = []
    honeypot_log: list[dict] = []
    removed_log: list[dict] = []
    removal_counter: Counter[str] = Counter()

    for record in iter_candidates_by_ids(candidates_path, id_set):
        cid = str(record["candidate_id"])
        meta = metadata.get(cid, {})

        exclude_hp, hp_rules, hp_details = _evaluate_honeypot(record, config)
        if exclude_hp:
            reason = _honeypot_reason_code(hp_rules)
            honeypot_log.append(
                {"candidate_id": cid, "rules": hp_rules, "details": hp_details}
            )
            removed_log.append({"candidate_id": cid, "reason_code": reason})
            removal_counter[reason] += 1
            continue

        exp = evaluate_experience(record, config)
        if exp.remove:
            removed_log.append({"candidate_id": cid, "reason_code": exp.reason or "exp_out_of_band"})
            removal_counter[exp.reason or "exp_out_of_band"] += 1
            continue

        title = evaluate_title(record, config)
        if title.remove:
            removed_log.append({"candidate_id": cid, "reason_code": title.reason or "non_eng_title"})
            removal_counter[title.reason or "non_eng_title"] += 1
            continue

        consulting = evaluate_consulting(record, config)
        if consulting.remove:
            removed_log.append(
                {"candidate_id": cid, "reason_code": consulting.reason or "consulting_only_career"}
            )
            removal_counter[consulting.reason or "consulting_only_career"] += 1
            continue

        research = evaluate_research(record, config)
        if research.remove:
            removed_log.append(
                {"candidate_id": cid, "reason_code": research.reason or "research_only_career"}
            )
            removal_counter[research.reason or "research_only_career"] += 1
            continue

        coding = evaluate_coding_recency(record, config)

        shallow = evaluate_shallow_ai(record, config)
        if shallow.remove:
            removed_log.append(
                {"candidate_id": cid, "reason_code": shallow.reason or "shallow_recent_ai_only"}
            )
            removal_counter[shallow.reason or "shallow_recent_ai_only"] += 1
            continue

        career = evaluate_career_shape(record, config)
        logistics = evaluate_logistics(record, config)
        validation = evaluate_external_validation(record)

        avail = evaluate_availability(record, config)

        cluster_rank = meta.get("cluster_rank")
        row: dict = {
            "candidate_id": cid,
            "cluster_id": meta.get("cluster_id"),
            "cluster_rank": cluster_rank if cluster_rank is not None else None,
            "dist_to_centroid": dist_map.get(cid),
            "total_years_exp": exp.total_years_exp,
            "exp_band": exp.exp_band,
            "in_sweet_spot": exp.in_sweet_spot,
            "title_family": title.title_family,
            "skill_kw_density": title.skill_kw_density,
            "title_ambiguous": title.title_ambiguous,
            "stale_profile": avail.stale_profile,
            "low_responder": avail.low_responder,
            "not_open": avail.not_open,
            "honeypot_anomaly_score": None,
            "product_company_count": consulting.product_company_count,
            "consulting_company_count": consulting.consulting_company_count,
            "product_company_fraction": consulting.product_company_fraction,
            "career_type": consulting.career_type,
            "research_fraction": research.research_fraction,
            "research_heavy": research.research_heavy,
            "has_any_production_role": research.has_any_production_role,
            "stale_coding": coding.stale_coding,
            "currently_between_roles": coding.currently_between_roles,
            "months_since_last_ic_role": coding.months_since_last_ic_role,
            "pre_llm_production_ml": shallow.pre_llm_production_ml,
            "recent_ai_only": shallow.recent_ai_only,
            "llm_framework_only": shallow.llm_framework_only,
            "ml_experience_start_year": shallow.ml_experience_start_year,
            "avg_tenure_per_employer": career.avg_tenure_per_employer,
            "short_hop_count": career.short_hop_count,
            "title_progression_jumps": career.title_progression_jumps,
            "location_tier": logistics.location_tier,
            "external_validation_score": validation.external_validation_score,
            "has_github": validation.has_github,
            "notice_period_days": avail.notice_period_days,
        }
        survivors.append(row)

    survivor_count = len(survivors)
    elapsed = perf_counter() - start

    survivors_df = pl.DataFrame(survivors) if survivors else pl.DataFrame()

    summary = {
        "input_count": input_count,
        "survivor_count": survivor_count,
        "removed_count": input_count - survivor_count,
        "removal_breakdown": dict(removal_counter),
        "elapsed_seconds": round(elapsed, 3),
    }

    write_stage2_outputs(output_dir, survivors_df, honeypot_log, removed_log, summary)
    _validate_counts(input_count, survivor_count, config)

    return Stage2Result(
        input_count=input_count,
        survivor_count=survivor_count,
        removal_counts=dict(removal_counter),
        elapsed_seconds=elapsed,
        output_dir=output_dir,
    )


def print_stage2_summary(result: Stage2Result) -> None:
    print("\n--- Stage 2 summary ---")
    print(f"Input:     {result.input_count:,}")
    print(f"Survivors: {result.survivor_count:,}")
    print(f"Removed:   {result.input_count - result.survivor_count:,}")
    print(f"Elapsed:   {result.elapsed_seconds:.2f}s")
    if result.removal_counts:
        print("\n--- Removal breakdown ---")
        for reason, count in sorted(result.removal_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"\nWrote outputs to {result.output_dir}")
