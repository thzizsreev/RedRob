"""Orchestrate stage 2–5 elimination collection and output writing."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eliminations_view.scripts.profiles import attach_profile, stream_candidates
from eliminations_view.scripts.stage2 import collect_stage2
from eliminations_view.scripts.stage3 import collect_stage3
from eliminations_view.scripts.stage4 import collect_stage4
from eliminations_view.scripts.stage5 import collect_stage5


def collect_eliminations(
    runtime_dir: Path,
    candidates_path: Path,
    *,
    stages: set[int] | None = None,
    honeypots_only: bool = False,
) -> dict[str, Any]:
    stage2_dir = runtime_dir / "stage2"
    stage3_dir = runtime_dir / "stage3"
    stage4_dir = runtime_dir / "stage4"
    stage5_dir = runtime_dir / "stage5"

    active = stages or {2, 3, 4, 5}
    by_stage: dict[int, list[dict[str, Any]]] = {}

    if 2 in active:
        by_stage[2] = collect_stage2(stage2_dir, honeypots_only=honeypots_only)
    if 3 in active:
        by_stage[3] = collect_stage3(stage2_dir, stage3_dir)
    if 4 in active:
        by_stage[4] = collect_stage4(stage3_dir, stage4_dir)
    if 5 in active:
        by_stage[5] = collect_stage5(stage4_dir, stage5_dir)

    all_records: list[dict[str, Any]] = []
    for stage in sorted(by_stage):
        all_records.extend(by_stage[stage])

    wanted = {r["candidate_id"] for r in all_records}
    profiles = stream_candidates(candidates_path, wanted) if wanted else {}

    missing = sorted(wanted - set(profiles))
    for record in all_records:
        attach_profile(record, profiles.get(record["candidate_id"]))

    counts_by_stage = {str(s): len(rows) for s, rows in by_stage.items()}
    reason_counts = Counter(
        r["elimination"]["reason_code"] for r in all_records
    )

    sources = {
        "runtime_dir": str(runtime_dir.resolve()),
        "candidates_jsonl": str(candidates_path.resolve()),
        "stages": sorted(active),
        "honeypots_only": honeypots_only,
    }

    return {
        "meta": {
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "elimination_count": len(all_records),
            "counts_by_stage": counts_by_stage,
            "reason_breakdown": dict(reason_counts),
            "missing_profiles": len(missing),
            "missing_profile_examples": missing[:10],
            "sources": sources,
        },
        "eliminations": all_records,
        "by_stage": by_stage,
    }


def write_outputs(dataset: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    all_json = out_dir / "all_eliminations.json"
    all_json.write_text(
        json.dumps(
            {"meta": dataset["meta"], "eliminations": dataset["eliminations"]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["all_json"] = str(all_json.resolve())

    for stage, rows in dataset["by_stage"].items():
        stage_dir = out_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_json = stage_dir / "eliminations.json"
        stage_json.write_text(
            json.dumps(
                {
                    "meta": {
                        **dataset["meta"],
                        "stage": stage,
                        "elimination_count": len(rows),
                    },
                    "eliminations": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        paths[f"stage{stage}_json"] = str(stage_json.resolve())

    return paths
