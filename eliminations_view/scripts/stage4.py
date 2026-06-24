"""Stage 4 elimination collector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eliminations_view.scripts._io import load_ids, load_json_dict, parquet_to_dict
from eliminations_view.scripts.reasons import build_elimination_meta

STAGE4_PIPELINE_KEYS = (
    "stage3_rank",
    "stage4_rank",
    "cross_encoder_score",
    "fused_score",
    "q1_score",
    "q1_rank",
    "q2_score",
    "q2_rank",
    "bm25_score",
    "bm25_rank",
    "rrf_score",
    "q3_neg_sim",
)


def collect_stage4(stage3_dir: Path, stage4_dir: Path) -> list[dict[str, Any]]:
    stage3_parquet = stage3_dir / "stage3_retrieved.parquet"
    stage3_json = stage3_dir / "stage3_retrieved.json"
    reranked_parquet = stage4_dir / "stage4_reranked.parquet"
    reranked_json = stage4_dir / "stage4_reranked.json"
    summary_path = stage4_dir / "stage4_summary.json"

    if stage3_parquet.exists():
        input_ids = set(load_ids(stage3_parquet))
        stage3_meta = parquet_to_dict(stage3_parquet)
    elif stage3_json.exists():
        input_ids = set(load_ids(stage3_json))
        stage3_meta = {}
    else:
        raise FileNotFoundError(f"Missing stage3 retrieved in {stage3_dir}")

    if reranked_parquet.exists():
        kept_ids = set(load_ids(reranked_parquet))
        reranked_meta = parquet_to_dict(reranked_parquet)
    elif reranked_json.exists():
        kept_ids = set(load_ids(reranked_json))
        reranked_meta = {}
    else:
        raise FileNotFoundError(f"Missing stage4 reranked in {stage4_dir}")

    summary = load_json_dict(summary_path) if summary_path.exists() else {}
    keep_n = summary.get("keep_n")

    eliminated_ids = sorted(input_ids - kept_ids)
    records: list[dict[str, Any]] = []

    for cid in eliminated_ids:
        source = reranked_meta.get(cid) or stage3_meta.get(cid, {})
        pipeline = {k: source.get(k) for k in STAGE4_PIPELINE_KEYS if k in source}
        if keep_n is not None:
            pipeline["keep_n"] = keep_n

        elimination = build_elimination_meta(
            "below_rerank_cutoff",
            pipeline=pipeline,
        )
        records.append(
            {
                "stage": 4,
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
