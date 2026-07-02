#!/usr/bin/env python3
"""Mandatory pre-submit gate — fail if traps in top-K."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranker import jd_config as jd
from ranker.honeypot import is_honeypot, soft_trap_flags
from ranker.submission_safe import is_submission_safe


def load_candidates(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                out[c["candidate_id"]] = c
    return out


def load_submission(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["candidate_id"])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="TRACER submit guard — zero traps in top-K.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    top_ids = load_submission(args.submission)[: args.top_k]

    hard_hits: list[str] = []
    soft_hits: list[str] = []
    negative_titles: list[str] = []

    for cid in top_ids:
        c = candidates.get(cid)
        if not c:
            hard_hits.append(f"{cid}:missing")
            continue
        hp, flags = is_honeypot(c)
        if hp:
            hard_hits.append(f"{cid}:{','.join(flags)}")
        title = c.get("profile", {}).get("current_title", "")
        if jd.TITLE_NEGATIVE.search(title):
            negative_titles.append(cid)
        soft = soft_trap_flags(c)
        if soft:
            soft_hits.append(f"{cid}:{','.join(soft)}")

    print(f"=== TRACER submit_guard top-{len(top_ids)} ===")
    print(f"Hard honeypots: {len(hard_hits)}")
    for h in hard_hits:
        print(f"  FAIL {h}")
    print(f"Negative titles: {len(negative_titles)}")
    for n in negative_titles:
        print(f"  FAIL {n}")
    print(f"Soft trap flags (informational): {len(soft_hits)}")

    if hard_hits or negative_titles:
        print("\nSUBMIT BLOCKED: fix traps before upload.", file=sys.stderr)
        return 1

    print("\nPASS: safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
