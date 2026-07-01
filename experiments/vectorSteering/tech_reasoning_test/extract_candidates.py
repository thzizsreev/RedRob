#!/usr/bin/env python3
"""Extract test candidates from team_results.json into input/candidates.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEAM_RESULTS = (
    ROOT
    / "outputs"
    / "team_views"
    / "Stage5ImpactCalibrated"
    / "team_results.json"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "input" / "candidates.json"

CANDIDATE_IDS = [
    "CAND_0018549",
    "CAND_0061265",
    "CAND_0086151",
    "CAND_0018722",
    "CAND_0051630",
]

PIPELINE_KEYS = (
    "retrieval_scores",
    "skill_assessment_scores",
    "gates_and_career",
    "behavioral_signals",
    "stage5_scoring",
)


def _slim_pipeline(pipeline: dict) -> dict:
    return {key: pipeline.get(key) for key in PIPELINE_KEYS if key in pipeline}


def extract(team_results_path: Path, output_path: Path) -> list[dict]:
    if not team_results_path.is_file():
        raise FileNotFoundError(
            f"team_results.json not found at {team_results_path}. "
            "Run: python tools/build_team_view.py --out outputs/team_views/Stage5ImpactCalibrated"
        )

    payload = json.loads(team_results_path.read_text(encoding="utf-8"))
    by_id = {c["candidate_id"]: c for c in payload.get("candidates", [])}

    missing = [cid for cid in CANDIDATE_IDS if cid not in by_id]
    if missing:
        raise KeyError(f"Candidates not found in team_results: {missing}")

    records: list[dict] = []
    for cid in CANDIDATE_IDS:
        source = by_id[cid]
        records.append(
            {
                "candidate_id": source["candidate_id"],
                "profile": source.get("profile") or {},
                "career_history": source.get("career_history") or [],
                "skills": source.get("skills") or [],
                "pipeline": _slim_pipeline(source.get("pipeline") or {}),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"candidates": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return records


def main() -> int:
    records = extract(TEAM_RESULTS, OUTPUT_PATH)
    print(f"Wrote {len(records)} candidates -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
