"""Stage 3 elimination collector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eliminations_view.scripts._io import load_csv_rows, load_ids, load_json_dict
from eliminations_view.scripts.reasons import build_elimination_meta

STAGE3_PIPELINE_KEYS = (
    "q1_score",
    "q1_rank",
    "q2_score",
    "q2_rank",
    "skill_score",
    "skill_rank",
    "rrf_score",
    "q3_neg_sim",
    "fused_score",
    "stage3_rank",
    "kept",
)


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() == "true"


def _load_score_distribution(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in load_csv_rows(path):
        cid = row["candidate_id"]
        pipeline: dict[str, Any] = {}
        for key in STAGE3_PIPELINE_KEYS:
            if key not in row:
                continue
            raw = row[key]
            if key == "kept":
                pipeline[key] = _parse_bool(raw)
            elif raw == "":
                pipeline[key] = None
            else:
                try:
                    if key.endswith("_rank"):
                        pipeline[key] = int(float(raw))
                    else:
                        pipeline[key] = float(raw)
                except ValueError:
                    pipeline[key] = raw
        by_id[cid] = pipeline
    return by_id


def collect_stage3(stage2_dir: Path, stage3_dir: Path) -> list[dict[str, Any]]:
    survivors_path = stage2_dir / "stage2_filtered_ids.json"
    retrieved_parquet = stage3_dir / "stage3_retrieved.parquet"
    retrieved_json = stage3_dir / "stage3_retrieved.json"
    distribution_path = stage3_dir / "stage3_score_distribution.csv"
    summary_path = stage3_dir / "stage3_summary.json"

    if not survivors_path.exists():
        raise FileNotFoundError(f"Missing {survivors_path}")

    survivor_ids = set(load_ids(survivors_path))
    if retrieved_parquet.exists():
        kept_ids = set(load_ids(retrieved_parquet))
    elif retrieved_json.exists():
        kept_ids = set(load_ids(retrieved_json))
    else:
        raise FileNotFoundError(f"Missing stage3 retrieved output in {stage3_dir}")

    eliminated_ids = sorted(survivor_ids - kept_ids)
    distribution = _load_score_distribution(distribution_path)
    summary = load_json_dict(summary_path) if summary_path.exists() else {}
    threshold = summary.get("threshold")

    records: list[dict[str, Any]] = []
    for cid in eliminated_ids:
        pipeline = dict(distribution.get(cid, {}))
        if threshold is not None:
            pipeline["threshold"] = threshold

        if cid not in distribution:
            reason_code = "not_in_retrieval_union"
        else:
            reason_code = "below_fused_cut"

        elimination = build_elimination_meta(
            reason_code,
            pipeline=pipeline,
        )
        records.append(
            {
                "stage": 3,
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
