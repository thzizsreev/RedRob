"""Stage 5 elimination collector."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from eliminations_view.scripts._io import load_ids, load_json_dict, parquet_to_dict
from eliminations_view.scripts.reasons import build_elimination_meta

STAGE5_PIPELINE_KEYS = (
    "stage3_rank",
    "stage4_rank",
    "cross_encoder_score",
    "fused_score",
    "final_score",
    "ce_norm",
    "fused_norm",
    "core",
    "penalized",
    "bonused",
    "total_penalty",
    "title_chasing_penalty",
    "q3_residual_penalty",
    "closed_source_penalty",
    "ambiguity_penalty",
    "consulting_resid_penalty",
    "availability_multiplier",
    "logistics_adjustment",
)


def collect_stage5(stage4_dir: Path, stage5_dir: Path) -> list[dict[str, Any]]:
    stage4_parquet = stage4_dir / "stage4_reranked.parquet"
    stage4_json = stage4_dir / "stage4_reranked.json"
    scored_parquet = stage5_dir / "stage5_scored.parquet"
    top_parquet = stage5_dir / "stage5_scored_top100.parquet"
    team_csv = stage5_dir / "team_xxx.csv"
    summary_path = stage5_dir / "stage5_summary.json"

    if stage4_parquet.exists():
        input_ids = set(load_ids(stage4_parquet))
    elif stage4_json.exists():
        input_ids = set(load_ids(stage4_json))
    else:
        raise FileNotFoundError(f"Missing stage4 reranked in {stage4_dir}")

    if top_parquet.exists():
        kept_ids = set(load_ids(top_parquet))
    elif team_csv.exists():
        with open(team_csv, encoding="utf-8", newline="") as f:
            kept_ids = {row["candidate_id"] for row in csv.DictReader(f)}
    else:
        raise FileNotFoundError(f"Missing stage5 top-N output in {stage5_dir}")

    scored_meta = parquet_to_dict(scored_parquet) if scored_parquet.exists() else {}
    summary = load_json_dict(summary_path) if summary_path.exists() else {}
    top_n = summary.get("output_count")

    eliminated_ids = sorted(input_ids - kept_ids)
    records: list[dict[str, Any]] = []

    for cid in eliminated_ids:
        source = scored_meta.get(cid, {})
        pipeline = {k: source.get(k) for k in STAGE5_PIPELINE_KEYS if k in source}
        if top_n is not None:
            pipeline["top_n"] = top_n

        elimination = build_elimination_meta(
            "below_final_cutoff",
            pipeline=pipeline,
        )
        records.append(
            {
                "stage": 5,
                "candidate_id": cid,
                "elimination": elimination,
                "pipeline": pipeline,
                "profile": {},
                "career_history": [],
                "education": [],
                "skills": [],
            }
        )

    return records
