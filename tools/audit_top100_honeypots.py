#!/usr/bin/env python3
"""
Audit top-K submission rows for honeypot signals (Stage 2 rules).

Hackathon Stage 3 disqualifies if honeypot rate > 10% in top 100.

    python tools/audit_top100_honeypots.py --submission SignalHunters.csv \\
        --candidates data/candidates.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage2.config import load_stage2_config
from tracks.instructor.stage2.checks.skills import evaluate_skill_honeypot
from tracks.instructor.stage2.honeypot_rules import evaluate_timeline_honeypot
from tracks.shared.paths import CANDIDATES_JSONL_PATH, ROOT_DIR

DEFAULT_FAIL_THRESHOLD = 0.10


def _load_candidates(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[str(record["candidate_id"])] = record
    return out


def _load_submission_ids(path: Path, top_k: int) -> list[str]:
    ids: list[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(str(row["candidate_id"]).strip())
            if len(ids) >= top_k:
                break
    return ids


def _evaluate_honeypot(record: dict, config_path: Path) -> tuple[bool, list[str]]:
    config = load_stage2_config(config_path)
    timeline = evaluate_timeline_honeypot(record, config)
    skills = evaluate_skill_honeypot(record, config)
    rules = timeline.rules_fired + skills.rules_fired
    exclude = timeline.exclude or skills.exclude
    return exclude, rules


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit submission top-K for honeypot patterns (Stage 2 rules)."
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=CANDIDATES_JSONL_PATH,
    )
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "config.yaml")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=DEFAULT_FAIL_THRESHOLD,
        help="Fail if hard honeypot rate exceeds this (default 0.10 = 10%%)",
    )
    args = parser.parse_args()

    if not args.submission.exists():
        print(f"Submission not found: {args.submission}", file=sys.stderr)
        return 1
    if not args.candidates.exists():
        print(f"Candidates not found: {args.candidates}", file=sys.stderr)
        return 1

    candidates = _load_candidates(args.candidates)
    top_ids = _load_submission_ids(args.submission, args.top_k)

    hits: list[tuple[str, list[str]]] = []
    missing = 0
    for cid in top_ids:
        record = candidates.get(cid)
        if record is None:
            missing += 1
            continue
        exclude, rules = _evaluate_honeypot(record, args.config)
        if exclude:
            hits.append((cid, rules))

    rate = len(hits) / max(len(top_ids), 1)
    print(f"Top-{len(top_ids)} honeypot flags (Stage 2 rules): {len(hits)} ({rate:.1%})")
    if missing:
        print(f"Warning: {missing} submission IDs not found in candidates file")
    for cid, rules in hits:
        print(f"  {cid}: {', '.join(rules) if rules else 'honeypot'}")

    top10 = top_ids[:10]
    top10_hits = [cid for cid, _ in hits if cid in top10]
    if top10_hits:
        print(f"Top-10 honeypot hits: {len(top10_hits)} — {', '.join(top10_hits)}")

    if rate > args.fail_threshold:
        print(
            f"FAIL: honeypot rate {rate:.1%} > {args.fail_threshold:.0%} "
            "(hackathon Stage 3 threshold)",
            file=sys.stderr,
        )
        return 1
    print("PASS: honeypot rate within threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
