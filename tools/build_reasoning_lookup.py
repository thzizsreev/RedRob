#!/usr/bin/env python3
"""
Build reasoning lookup from a submission CSV.

    python tools/build_reasoning_lookup.py artifacts/runtime/stage6/SignalHunters.csv

Output: artifacts/precomputed/reasoning_lookup.json
  { "by_candidate_id": { "CAND_...": { "rank", "score", "reasoning" } } }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage6.io import export_reasoning_lookup_from_csv
from tracks.shared.paths import REASONING_LOOKUP_PATH, TEAM_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reasoning from submission CSV into lookup JSON."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=None,
        help="Submission CSV (default: artifacts/runtime/stage6/SignalHunters.csv from config)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REASONING_LOOKUP_PATH,
        help="Output lookup JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path
    if csv_path is None:
        import yaml

        cfg_path = _ROOT / "config.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        team_id = raw.get("stage6", {}).get("team_id", TEAM_ID)
        out_dir = raw.get("stage6", {}).get("output_dir", "artifacts/runtime/stage6")
        csv_path = (_ROOT / out_dir / f"{team_id}.csv").resolve()
    else:
        csv_path = csv_path.resolve()
    out = export_reasoning_lookup_from_csv(csv_path, args.out.resolve())
    print(f"Reasoning lookup ready: {out}")


if __name__ == "__main__":
    main()
