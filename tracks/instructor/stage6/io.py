"""Stage 6 I/O — load top-100, join profiles, write CSV and audit parquet."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import polars as pl

from tracks.instructor.core.io import iter_candidates_from_path
from tracks.instructor.stage6.assemble import assemble_candidate_dict
from tracks.instructor.stage6.config import Stage6Config
from tracks.shared.paths import REASONING_LOOKUP_PATH


def load_stage5_top100(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 5 top-100 parquet: {path}")
    df = pl.read_parquet(path)
    if "candidate_id" not in df.columns:
        raise ValueError(f"{path} missing candidate_id column")
    if "rank" not in df.columns:
        df = df.with_row_index("rank", offset=1)
    print(f"Stage 5 top-100 loaded: {df.height:,} rows from {path.name}")
    return df


def load_jsonl_records(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    wanted = set(candidate_ids)
    found: dict[str, dict[str, Any]] = {}
    for record in iter_candidates_from_path(path):
        cid = str(record.get("candidate_id", ""))
        if cid not in wanted or cid in found:
            continue
        found[cid] = record
        if len(found) == len(wanted):
            break

    missing = wanted - set(found.keys())
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} top-100 candidate(s) missing from {path}. Examples: {examples}"
        )
    print(f"Joined JSONL profiles for {len(found):,} candidates")
    return found


def load_skill_scores(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    if "candidate_id" not in df.columns:
        return {}

    skill_cols = [
        c
        for c in df.columns
        if c != "candidate_id" and df[c].dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32)
    ]
    if not skill_cols:
        return {}

    out: dict[str, dict[str, float]] = {}
    for row in df.iter_rows(named=True):
        cid = str(row["candidate_id"])
        if cid not in candidate_ids:
            continue
        scores = {col: float(row[col]) for col in skill_cols if row.get(col) is not None}
        if scores:
            out[cid] = scores
    print(f"Loaded skill assessment scores for {len(out):,} candidates")
    return out


def load_reasoning_raw_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        print(f"No reasoning raw cache at {path} — building all sentences at runtime.")
        return {}

    df = pl.read_parquet(path)
    required = {"candidate_id", "s1_raw", "s2_raw", "tech_cat"}
    if not required.issubset(set(df.columns)):
        print(f"Reasoning raw cache missing columns {required - set(df.columns)} — ignoring.")
        return {}

    cache: dict[str, dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        cid = str(row["candidate_id"])
        cache[cid] = dict(row)
    print(f"Loaded reasoning raw cache for {len(cache):,} candidates")
    return cache


def parse_ranking_csv(path: Path) -> list[dict[str, Any]]:
    """Read ranking-only CSV rows (candidate_id, rank, score)."""
    if not path.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames not in (
            ["candidate_id", "rank", "score"],
            ["candidate_id", "rank", "score", "reasoning"],
        ):
            raise ValueError(
                f"Unexpected ranking CSV header in {path}: {reader.fieldnames}"
            )
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "candidate_id": str(row["candidate_id"]).strip(),
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                }
            )
    return rows


def parse_submission_csv(path: Path) -> list[dict[str, Any]]:
    """Read submission CSV rows (candidate_id, rank, score, reasoning)."""
    if not path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["candidate_id", "rank", "score", "reasoning"]:
            raise ValueError(
                f"Unexpected CSV header in {path}: {reader.fieldnames}"
            )
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "candidate_id": str(row["candidate_id"]).strip(),
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "reasoning": str(row["reasoning"]),
                }
            )
    return rows


def build_reasoning_lookup(
    submission_rows: list[dict[str, Any]],
    *,
    source: str,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Build lookup payload keyed by candidate_id."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in submission_rows:
        cid = str(row["candidate_id"])
        by_id[cid] = {
            "rank": int(row["rank"]),
            "score": float(row["score"]),
            "reasoning": str(row["reasoning"]),
        }
    return {
        "source": source,
        "source_path": str(source_path.resolve()) if source_path else None,
        "row_count": len(by_id),
        "by_candidate_id": by_id,
    }


def write_reasoning_lookup(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote reasoning lookup ({payload.get('row_count', 0)} rows) -> {path}")
    return path


def load_reasoning_lookup(path: Path) -> dict[str, dict[str, Any]]:
    """Return by_candidate_id map, or empty dict if missing/invalid."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read reasoning lookup at {path}: {exc}")
        return {}
    by_id = payload.get("by_candidate_id")
    if not isinstance(by_id, dict):
        print(f"Reasoning lookup at {path} missing by_candidate_id — ignoring.")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for cid, entry in by_id.items():
        if not isinstance(entry, dict) or "reasoning" not in entry:
            continue
        out[str(cid)] = dict(entry)
    print(f"Loaded reasoning lookup for {len(out):,} candidates from {path.name}")
    return out


def reasoning_rows_from_lookup(
    candidate_ids: list[str],
    lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Build minimal reasoning rows when every candidate_id is in the lookup."""
    missing = [cid for cid in candidate_ids if cid not in lookup]
    if missing:
        return None
    return [
        {
            "candidate_id": cid,
            "reasoning": str(lookup[cid]["reasoning"]),
            "from_lookup": True,
        }
        for cid in candidate_ids
    ]


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


def write_stage6_outputs(
    output_dir: Path,
    team_id: str,
    submission_rows: list[dict],
    reasoning_rows: list[dict],
    summary: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{team_id}.csv"
    write_submission_csv(csv_path, submission_rows)

    if reasoning_rows:
        pl.DataFrame(reasoning_rows).write_parquet(output_dir / "stage6_reasoning.parquet")

    with open(output_dir / "stage6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'stage6_reasoning.parquet'}")
    print(f"Wrote {output_dir / 'stage6_summary.json'}")
    return csv_path


def write_reasoning_lookup_from_submission(
    submission_rows: list[dict],
    csv_path: Path,
    lookup_path: Path | None = None,
) -> Path:
    """Persist candidate_id -> rank, score, reasoning for fast Stage 6 re-runs."""
    out_path = lookup_path or REASONING_LOOKUP_PATH
    payload = build_reasoning_lookup(
        submission_rows,
        source="stage6_submission_csv",
        source_path=csv_path,
    )
    return write_reasoning_lookup(out_path, payload)


def export_reasoning_lookup_from_csv(
    csv_path: Path,
    lookup_path: Path | None = None,
) -> Path:
    """Build lookup artifact from an existing submission CSV."""
    rows = parse_submission_csv(csv_path)
    out_path = lookup_path or REASONING_LOOKUP_PATH
    payload = build_reasoning_lookup(
        rows,
        source="submission_csv",
        source_path=csv_path,
    )
    return write_reasoning_lookup(out_path, payload)


def build_candidate_dicts_for_ranking(
    config: Stage6Config,
    ranking_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sorted_rows = sorted(ranking_rows, key=lambda r: int(r["rank"]))
    ids = {str(r["candidate_id"]) for r in sorted_rows}
    top_df = load_stage5_top100(config.stage5_top100_path)
    by_id = {
        str(row["candidate_id"]): dict(row) for row in top_df.iter_rows(named=True)
    }
    missing = ids - set(by_id)
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} ranking CSV candidate(s) missing from Stage 5 parquet. "
            f"Examples: {examples}"
        )
    jsonl = load_jsonl_records(config.candidates_jsonl_path, ids)
    skills = load_skill_scores(config.candidate_features_path, ids)
    candidates: list[dict[str, Any]] = []
    for row in sorted_rows:
        cid = str(row["candidate_id"])
        candidates.append(
            assemble_candidate_dict(
                by_id[cid],
                jsonl[cid],
                skill_scores=skills.get(cid),
            )
        )
    return candidates


def build_candidate_dicts(config: Stage6Config) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    top_df = load_stage5_top100(config.stage5_top100_path)
    ids = set(top_df["candidate_id"].cast(pl.Utf8).to_list())
    jsonl = load_jsonl_records(config.candidates_jsonl_path, ids)
    skills = load_skill_scores(config.candidate_features_path, ids)

    candidates: list[dict[str, Any]] = []
    for row in top_df.sort("rank").iter_rows(named=True):
        cid = str(row["candidate_id"])
        candidates.append(
            assemble_candidate_dict(
                dict(row),
                jsonl[cid],
                skill_scores=skills.get(cid),
            )
        )
    return top_df, candidates
