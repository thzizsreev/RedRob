"""Stage 4 I/O — load Stage 3, resolve candidate text, write outputs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl

from tracks.instructor.core.extraction import build_candidate_passage
from tracks.instructor.core.io import iter_candidates_from_path
from tracks.instructor.stage4.config import Stage4Config

REQUIRED_STAGE3_COLUMNS = frozenset(
    {
        "candidate_id",
        "stage3_rank",
        "fused_score",
        "q1_score",
        "q2_score",
        "skill_score",
        "q3_neg_sim",
        "rrf_score",
        "exp_band",
        "in_sweet_spot",
        "title_family",
        "skill_kw_density",
        "title_ambiguous",
        "stale_profile",
        "low_responder",
        "not_open",
    }
)


def load_stage3_retrieved(path: Path, config: Stage4Config) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 3 output: {path}")

    df = pl.read_parquet(path)
    missing = REQUIRED_STAGE3_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"stage3_retrieved.parquet missing required columns: {sorted(missing)}")

    count = df.height
    print(f"Stage 3 retrieved loaded: {count:,}")
    if count < config.expected_input_min or count > config.expected_input_max:
        warnings.warn(
            f"Stage 3 row count {count} outside expected range "
            f"[{config.expected_input_min}, {config.expected_input_max}]",
            stacklevel=2,
        )
    return df


def _load_summaries_from_features(
    features_path: Path,
    candidate_ids: set[str],
) -> dict[str, str]:
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


def _load_passages_from_jsonl(
    candidates_path: Path,
    candidate_ids: set[str],
) -> dict[str, str]:
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    passages: dict[str, str] = {}
    for record in iter_candidates_from_path(candidates_path):
        cid = record.get("candidate_id")
        if cid is None or cid not in candidate_ids or cid in passages:
            continue
        passages[cid] = build_candidate_passage(record)

    missing = candidate_ids - set(passages.keys())
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} Stage 3 candidate(s) missing from {candidates_path}. "
            f"Examples: {examples}"
        )
    print(f"Loaded {len(passages):,} candidate passages from {candidates_path.name}")
    return passages


def load_skills_for_ids(
    candidates_path: Path,
    candidate_ids: set[str],
) -> dict[str, list[dict]]:
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    wanted = set(candidate_ids)
    found: dict[str, list[dict]] = {}
    for record in iter_candidates_from_path(candidates_path):
        cid = record.get("candidate_id")
        if cid is None or cid not in wanted or cid in found:
            continue
        found[str(cid)] = list(record.get("skills") or [])
        if len(found) == len(wanted):
            break

    missing = wanted - set(found.keys())
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} candidate(s) missing from {candidates_path}. "
            f"Examples: {examples}"
        )
    print(f"Loaded skills for {len(found):,} candidates from {candidates_path.name}")
    return found


def resolve_candidate_texts(
    candidate_ids: list[str],
    config: Stage4Config,
) -> dict[str, str]:
    id_set = set(candidate_ids)
    summaries = _load_summaries_from_features(config.candidate_features_path, id_set)
    need_passages = id_set - set(summaries.keys())
    passages = (
        _load_passages_from_jsonl(config.candidates_jsonl_path, need_passages)
        if need_passages
        else {}
    )

    text_by_id: dict[str, str] = dict(summaries)
    text_by_id.update(passages)
    return text_by_id


def _write_json(path: Path, df: pl.DataFrame) -> None:
    records = df.to_dicts() if df.height > 0 else []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def write_stage4_outputs(
    output_dir: Path,
    reranked_df: pl.DataFrame,
    rank_delta_df: pl.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    reranked_df.write_parquet(output_dir / "stage4_reranked.parquet")
    _write_json(output_dir / "stage4_reranked.json", reranked_df)
    rank_delta_df.write_csv(output_dir / "stage4_rank_delta.csv")

    with open(output_dir / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {output_dir / 'stage4_reranked.parquet'}")
    print(f"Wrote {output_dir / 'stage4_reranked.json'}")
    print(f"Wrote {output_dir / 'stage4_rank_delta.csv'}")
    print(f"Wrote {output_dir / 'stage4_summary.json'}")
