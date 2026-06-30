#!/usr/bin/env python3
"""
Join Stage 4 reranked candidates with full profiles from candidates.jsonl.

    python build_collection.py
    python build_collection.py --out ../stage4_reranked_collection

Default output: ../stage4_reranked_collection/ (outside this repo)
  collection.json, collection.jsonl, summary.csv, manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.shared.paths import CANDIDATES_JSONL_PATH, ROOT_DIR, RUNTIME_STAGE4_DIR, TEST_RUNS_DIR

DEFAULT_OUT_DIR = ROOT_DIR.parent / "stage4_reranked_collection"
STAGE4_JSON = RUNTIME_STAGE4_DIR / "stage4_reranked.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage 4 + full-profile analysis collection."
    )
    parser.add_argument(
        "--stage4",
        type=Path,
        default=STAGE4_JSON,
        help="Stage 4 reranked JSON input",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=CANDIDATES_JSONL_PATH,
        help="Candidates JSONL source",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for collection artifacts",
    )
    return parser.parse_args()


def load_stage4(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["candidate_id"]): r for r in rows}


def stream_candidates(path: Path, wanted: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    remaining = set(wanted)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = str(record.get("candidate_id", ""))
            if cid in remaining:
                found[cid] = record
                remaining.remove(cid)
    return found


def flatten_summary_row(merged: dict) -> dict:
    profile = merged.get("profile") or {}
    signals = merged.get("redrob_signals") or {}
    ranking = merged.get("ranking") or {}

    skills = merged.get("skills") or []
    skill_names = []
    for s in skills:
        if isinstance(s, dict):
            skill_names.append(str(s.get("name", s.get("skill", ""))))
        else:
            skill_names.append(str(s))

    current_title = profile.get("current_title", "")
    if not current_title and merged.get("career_history"):
        first = merged["career_history"][0]
        if isinstance(first, dict):
            current_title = first.get("title", "")

    return {
        "candidate_id": merged.get("candidate_id", ""),
        "stage4_rank": ranking.get("stage4_rank"),
        "stage3_rank": ranking.get("stage3_rank"),
        "cross_encoder_score": ranking.get("cross_encoder_score"),
        "fused_score": ranking.get("fused_score"),
        "q1_score": ranking.get("q1_score"),
        "q2_score": ranking.get("q2_score"),
        "skill_score": ranking.get("skill_score"),
        "rrf_score": ranking.get("rrf_score"),
        "total_years_exp": ranking.get("total_years_exp"),
        "exp_band": ranking.get("exp_band"),
        "in_sweet_spot": ranking.get("in_sweet_spot"),
        "title_family": ranking.get("title_family"),
        "skill_kw_density": ranking.get("skill_kw_density"),
        "anonymized_name": profile.get("anonymized_name", ""),
        "headline": profile.get("headline", ""),
        "current_title": current_title,
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "summary": profile.get("summary", ""),
        "skills_top": "; ".join(skill_names[:15]),
        "open_to_work": signals.get("open_to_work_flag"),
        "profile_completeness": signals.get("profile_completeness_score"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "career_roles_count": len(merged.get("career_history") or []),
        "education_count": len(merged.get("education") or []),
    }


def merge_record(ranking: dict, candidate: dict) -> dict:
    merged = dict(candidate)
    merged["ranking"] = ranking
    return merged


def main() -> None:
    args = parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.stage4.exists():
        raise FileNotFoundError(f"Missing Stage 4 output: {args.stage4}")
    if not args.candidates.exists():
        raise FileNotFoundError(f"Missing candidates file: {args.candidates}")

    rankings_by_id = load_stage4(args.stage4)
    wanted = set(rankings_by_id)
    profiles_by_id = stream_candidates(args.candidates, wanted)

    missing = sorted(wanted - set(profiles_by_id))
    if missing:
        raise ValueError(
            f"{len(missing)} Stage 4 candidate(s) not found in JSONL. "
            f"Examples: {missing[:5]}"
        )

    collection = [
        merge_record(rankings_by_id[cid], profiles_by_id[cid])
        for cid in sorted(
            rankings_by_id,
            key=lambda c: rankings_by_id[c].get("stage4_rank", 9999),
        )
    ]

    collection_json = out_dir / "collection.json"
    collection_jsonl = out_dir / "collection.jsonl"
    summary_csv = out_dir / "summary.csv"
    manifest_json = out_dir / "manifest.json"

    collection_json.write_text(
        json.dumps(collection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with open(collection_jsonl, "w", encoding="utf-8") as f:
        for row in collection:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = [flatten_summary_row(r) for r in collection]
    if summary_rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(collection),
        "sources": {
            "stage4_reranked": str(args.stage4.resolve()),
            "candidates_jsonl": str(args.candidates.resolve()),
        },
        "outputs": {
            "collection_json": str(collection_json.resolve()),
            "collection_jsonl": str(collection_jsonl.resolve()),
            "summary_csv": str(summary_csv.resolve()),
        },
        "stage4_rank_range": [
            collection[0]["ranking"]["stage4_rank"],
            collection[-1]["ranking"]["stage4_rank"],
        ],
        "missing_profiles": missing,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Built {len(collection)} merged profiles")
    print(f"  {collection_json}")
    print(f"  {collection_jsonl}")
    print(f"  {summary_csv}")
    print(f"  {manifest_json}")


if __name__ == "__main__":
    main()
