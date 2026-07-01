"""Stage 0 — pool-wide raw sentence precompute (s1/s2, no paraphrase)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

from tracks.instructor.core.io import iter_candidates_from_path
from tracks.instructor.stage6.assemble import assemble_candidate_dict
from tracks.instructor.stage6.reasoning_builder import build_raw_sentences, select_verb
from tracks.shared.paths import CANDIDATES_JSONL_PATH, PRECOMPUTED_DIR, REASONING_RAW_PATH, ROOT_DIR


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("stage6", {})


def _minimal_stage5_row(record: dict[str, Any]) -> dict[str, Any]:
    signals = record.get("redrob_signals") or {}
    profile = record.get("profile") or {}
    return {
        "candidate_id": record["candidate_id"],
        "cross_encoder_score": 0.0,
        "total_years_exp": profile.get("years_of_experience", 0),
        "in_sweet_spot": False,
        "pre_llm_production_ml": False,
        "product_company_fraction": 0.0,
        "consulting_company_count": 0,
        "avg_tenure_per_employer": 0.0,
        "llm_framework_only": False,
        "recent_ai_only": False,
        "days_since_active": 0,
        "open_to_work_flag": signals.get("open_to_work_flag", False),
        "applications_submitted_30d": signals.get("applications_submitted_30d", 0),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", 0),
        "github_activity_score": signals.get("github_activity_score", -1),
        "notice_period_days": signals.get("notice_period_days", 0),
    }


def run_reasoning_raw_precompute(
    *,
    candidates_path: Path,
    output_path: Path,
    candidate_features_path: Path | None = None,
    limit: int | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skill_map: dict[str, dict[str, float]] = {}
    if candidate_features_path and candidate_features_path.exists():
        df = pl.read_parquet(candidate_features_path)
        skill_cols = [
            c
            for c in df.columns
            if c != "candidate_id" and df[c].dtype in (pl.Float32, pl.Float64, pl.Int64)
        ]
        for row in df.iter_rows(named=True):
            cid = str(row["candidate_id"])
            scores = {col: float(row[col]) for col in skill_cols if row.get(col) is not None}
            if scores:
                skill_map[cid] = scores

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(iter_candidates_from_path(candidates_path)):
        if limit is not None and idx >= limit:
            break
        cid = str(record["candidate_id"])
        candidate = assemble_candidate_dict(
            _minimal_stage5_row(record),
            record,
            skill_scores=skill_map.get(cid),
        )
        raw = build_raw_sentences(candidate)
        rows.append(
            {
                "candidate_id": raw["candidate_id"],
                "tech_cat": raw["tech_cat"],
                "s1_raw": raw["s1_raw"],
                "s2_raw": raw["s2_raw"],
                "temperature_s1": raw["temperature_s1"],
                "temperature_s2": raw["temperature_s2"],
                "temperature_s3": raw["temperature_s3"],
                "verb": select_verb(cid),
            }
        )
        if (idx + 1) % 5000 == 0:
            print(f"  precomputed {idx + 1:,} candidates...")

    pl.DataFrame(rows).write_parquet(output_path)
    print(f"Wrote {len(rows):,} rows -> {output_path}")
    return output_path


def run_from_config(config_path: Path) -> Path:
    s6 = _load_config(config_path)
    candidates_path = _resolve_path(str(s6.get("candidates_jsonl_path", CANDIDATES_JSONL_PATH)))
    output_path = _resolve_path(str(s6.get("reasoning_raw_path", REASONING_RAW_PATH)))
    features_path = _resolve_path(
        str(s6.get("candidate_features_path", PRECOMPUTED_DIR / "candidate_features.parquet"))
    )
    return run_reasoning_raw_precompute(
        candidates_path=candidates_path,
        output_path=output_path,
        candidate_features_path=features_path,
    )
