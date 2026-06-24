"""Stage 2 elimination collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eliminations_view.scripts._io import load_csv_rows
from eliminations_view.scripts.reasons import build_elimination_meta, parse_honeypot_rules


def collect_stage2(stage2_dir: Path, *, honeypots_only: bool = False) -> list[dict[str, Any]]:
    removed_path = stage2_dir / "stage2_removed_log.csv"
    honeypot_path = stage2_dir / "stage2_honeypot_log.csv"

    if not removed_path.exists():
        raise FileNotFoundError(f"Missing {removed_path}")

    honeypot_by_id: dict[str, dict[str, Any]] = {}
    if honeypot_path.exists():
        for row in load_csv_rows(honeypot_path):
            cid = row["candidate_id"]
            rules = [r for r in row.get("rules", "").split("|") if r]
            details_raw = row.get("details_json", "{}")
            try:
                details = json.loads(details_raw) if details_raw else {}
            except json.JSONDecodeError:
                details = {}
            honeypot_by_id[cid] = {"rules": rules, "details": details}

    records: list[dict[str, Any]] = []
    for row in load_csv_rows(removed_path):
        cid = row["candidate_id"]
        reason_code = row["reason_code"]
        if honeypots_only and not reason_code.startswith("honeypot_"):
            continue

        hp = honeypot_by_id.get(cid, {})
        rules = parse_honeypot_rules(reason_code, hp.get("rules"))
        details = hp.get("details", {})
        elimination = build_elimination_meta(reason_code, rules=rules, details=details)

        records.append(
            {
                "stage": 2,
                "candidate_id": cid,
                "elimination": elimination,
                "pipeline": {},
                "profile": {},
                "career_history": [],
                "education": [],
                "skills": [],
            }
        )

    return records
