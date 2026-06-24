"""Stage 5 I/O — load Stage 4, join signals, write outputs."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import polars as pl

from tracks.instructor.core.io import iter_candidates_from_path
from tracks.instructor.stage5.config import Stage5Config

REQUIRED_STAGE4_COLUMNS = frozenset(
    {
        "candidate_id",
        "cross_encoder_score",
        "fused_score",
        "q1_score",
        "q2_score",
        "q3_neg_sim",
        "exp_band",
        "in_sweet_spot",
        "product_company_fraction",
        "career_type",
        "stale_coding",
        "has_any_production_role",
        "short_hop_count",
        "title_ambiguous",
        "external_validation_score",
        "has_github",
        "location_tier",
        "notice_period_days",
    }
)


def load_stage4_reranked(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 4 output: {path}")
    df = pl.read_parquet(path)
    missing = REQUIRED_STAGE4_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"stage4_reranked.parquet missing required columns: {sorted(missing)}")
    print(f"Stage 4 reranked loaded: {df.height:,}")
    return df


def _load_summaries(features_path: Path, candidate_ids: set[str]) -> dict[str, str]:
    if not features_path.exists():
        return {}
    df = pl.read_parquet(features_path)
    if "technical_summary_sentence" not in df.columns:
        return {}
    summaries: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        cid = str(row["candidate_id"])
        if cid not in candidate_ids:
            continue
        summary = row.get("technical_summary_sentence")
        if summary is not None and str(summary).strip():
            summaries[cid] = str(summary).strip()
    print(f"Loaded {len(summaries):,} technical summaries from {features_path.name}")
    return summaries


def _extract_join_fields(record: dict) -> dict:
    profile = record.get("profile") or {}
    signals = record.get("redrob_signals") or {}
    assessments = signals.get("skill_assessment_scores") or {}
    return {
        "skills": record.get("skills") or [],
        "skill_assessment_scores": assessments,
        "last_active_date": signals.get("last_active_date"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "applications_submitted_30d": signals.get("applications_submitted_30d"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "avg_response_time_hours": signals.get("avg_response_time_hours"),
        "interview_completion_rate": signals.get("interview_completion_rate"),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate"),
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "profile_headline": profile.get("headline"),
    }


def load_candidate_join_data(
    candidates_path: Path,
    candidate_ids: set[str],
) -> dict[str, dict]:
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    wanted = set(candidate_ids)
    found: dict[str, dict] = {}
    for record in iter_candidates_from_path(candidates_path):
        cid = str(record.get("candidate_id", ""))
        if cid not in wanted or cid in found:
            continue
        found[cid] = _extract_join_fields(record)
        if len(found) == len(wanted):
            break

    missing = wanted - set(found.keys())
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} Stage 4 candidate(s) missing from {candidates_path}. "
            f"Examples: {examples}"
        )
    print(f"Joined behavioral/skills data for {len(found):,} candidates")
    return found


def join_scoring_inputs(
    stage4_df: pl.DataFrame,
    config: Stage5Config,
) -> pl.DataFrame:
    ids = set(stage4_df["candidate_id"].cast(pl.Utf8).to_list())
    join_data = load_candidate_join_data(config.candidates_jsonl_path, ids)
    summaries = _load_summaries(config.candidate_features_path, ids)

    join_rows = []
    for cid in stage4_df["candidate_id"].cast(pl.Utf8).to_list():
        row = dict(join_data[cid])
        row["technical_summary_sentence"] = summaries.get(cid) or row.get("profile_headline")
        join_rows.append(row)

    join_df = pl.DataFrame(join_rows)
    return pl.concat([stage4_df, join_df], how="horizontal")


def write_submission_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_stage5_outputs(
    output_dir: Path,
    team_id: str,
    scored_df: pl.DataFrame,
    top_df: pl.DataFrame,
    submission_rows: list[dict],
    summary: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{team_id}.csv"
    write_submission_csv(csv_path, submission_rows)

    scored_df.write_parquet(output_dir / "stage5_scored.parquet")
    top_df.write_parquet(output_dir / "stage5_scored_top100.parquet")

    with open(output_dir / "stage5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'stage5_scored.parquet'}")
    print(f"Wrote {output_dir / 'stage5_scored_top100.parquet'}")
    print(f"Wrote {output_dir / 'stage5_summary.json'}")
    return csv_path


def warn_input_count(count: int) -> None:
    if count < 250 or count > 350:
        warnings.warn(
            f"Stage 5 input count {count} outside expected ~300 (250-350)",
            stacklevel=2,
        )
