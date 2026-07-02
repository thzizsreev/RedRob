"""Stage 5 elimination collector."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eliminations_view.scripts._io import load_ids, load_json_dict, parquet_to_dict
from eliminations_view.scripts.reasons import build_elimination_meta
from tracks.shared.paths import TEAM_ID

STAGE5_PIPELINE_KEYS = (
    "stage3_rank",
    "stage4_rank",
    "cross_encoder_score",
    "fused_score",
    "final_score",
    "borda_primary",
    "borda_sum",
    "t1_std",
    "tier2_raw",
    "tier2_scaled",
    "avail_tier",
    "avail_unit",
    "tier3_scaled",
    "tier4_scaled",
    "days_since_active",
    "title_chasing_penalty",
    "optional_bonus",
)


def collect_stage5(stage4_dir: Path, stage5_dir: Path) -> list[dict[str, Any]]:
    stage4_parquet = stage4_dir / "stage4_reranked.parquet"
    stage4_json = stage4_dir / "stage4_reranked.json"
    scored_parquet = stage5_dir / "stage5_scored.parquet"
    top_parquet = stage5_dir / "stage5_scored_top100.parquet"
    team_csv = stage5_dir / f"{TEAM_ID}.csv"
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
