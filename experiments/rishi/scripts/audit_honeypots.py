#!/usr/bin/env python3
"""Audit submission for honeypot/trap patterns in top-K."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ranker.honeypot import is_honeypot, soft_trap_flags


def load_candidates(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            out[c["candidate_id"]] = c
    return out


def load_submission(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["candidate_id"])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit top-K for honeypot patterns.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "candidates.jsonl")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--fail-threshold", type=float, default=0.05)
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    top_ids = load_submission(args.submission)[: args.top_k]

    hard_hits: list[tuple[str, list[str]]] = []
    soft_hits: list[tuple[str, list[str]]] = []
    for cid in top_ids:
        c = candidates.get(cid)
        if not c:
            continue
        hp, flags = is_honeypot(c)
        if hp:
            hard_hits.append((cid, flags))
        soft = soft_trap_flags(c)
        if soft:
            soft_hits.append((cid, soft))

    rate = len(hard_hits) / max(len(top_ids), 1)
    print(f"Top-{len(top_ids)} hard honeypots: {len(hard_hits)} ({rate:.1%})")
    for cid, flags in hard_hits:
        print(f"  HARD {cid}: {', '.join(flags)}")
    print(f"Top-{len(top_ids)} soft trap flags: {len(soft_hits)}")
    for cid, flags in soft_hits[:10]:
        print(f"  SOFT {cid}: {', '.join(flags)}")
    if len(soft_hits) > 10:
        print(f"  ... and {len(soft_hits) - 10} more soft flags")

    if rate > args.fail_threshold:
        print(f"FAIL: honeypot rate {rate:.1%} > {args.fail_threshold:.0%}", file=sys.stderr)
        return 1
    print("PASS: honeypot rate within threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
