#!/usr/bin/env python3
"""Verify all top-100 submission candidates against JD fit and trap rules."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ranker import jd_config as jd
from ranker.honeypot import is_honeypot, soft_trap_flags
from ranker.io import rank_candidates
from ranker.scorer import score_candidate


def load_candidates(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                out[c["candidate_id"]] = c
    return out


def load_submission(path: Path) -> list[tuple[int, str, float, str]]:
    rows: list[tuple[int, str, float, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["rank"]), row["candidate_id"], float(row["score"]), row["reasoning"]))
    return sorted(rows, key=lambda x: x[0])


def title_tier(title: str) -> str:
    if jd.TITLE_TIER1.search(title):
        return "tier1"
    if jd.TITLE_TIER2.search(title):
        return "tier2"
    if jd.TITLE_NEGATIVE.search(title):
        return "negative"
    return "other"


def classify(c: dict, ctx: dict) -> tuple[str, list[str]]:
    f = ctx["features"]
    concerns: list[str] = []
    p = c.get("profile", {})
    title = p.get("current_title", "")
    country = (p.get("country") or "").lower()
    yoe = float(p.get("years_of_experience", 0))
    sem = ctx.get("semantic_score", 0)
    tt = title_tier(title)

    if is_honeypot(c)[0]:
        return "REJECT", ["hard_honeypot"]
    if tt == "negative":
        return "REJECT", ["negative_title"]
    if soft_trap_flags(c):
        concerns.extend(soft_trap_flags(c))
    if country and country != "india":
        concerns.append(f"non_india:{country}")
    if yoe < 4 or yoe > 12:
        concerns.append(f"yoe_edge:{yoe}")
    if sem < 0.55:
        concerns.append(f"low_semantic:{sem:.2f}")
    if f["title_score"] < 0.2:
        concerns.append("weak_title")
    if f.get("penalty_reasons"):
        concerns.extend(f["penalty_reasons"])

    if tt in ("tier1", "tier2") and sem >= 0.65 and f["career_score"] >= 0.25 and yoe >= 4:
        return "STRONG", concerns
    if tt in ("tier1", "tier2") and sem >= 0.55:
        return "GOOD", concerns
    if sem >= 0.60 and f["career_score"] >= 0.2:
        return "GOOD", concerns
    return "WEAK", concerns


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=PROJECT_ROOT / "submission.csv")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "candidates.jsonl")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--json-out", type=Path, default=ROOT / "verify_top100_report.json")
    parser.add_argument("--time-rank", action="store_true")
    args = parser.parse_args()

    import numpy as np

    ids_order = json.loads((args.artifacts / "candidate_ids.json").read_text())
    sem_arr = np.load(args.artifacts / "semantic_scores.npy")
    sem_map = {cid: float(sem_arr[i]) for i, cid in enumerate(ids_order)}

    candidates = load_candidates(args.candidates)
    submission = load_submission(args.submission)

    rows: list[dict] = []
    for rank, cid, sub_score, reasoning in submission:
        c = candidates[cid]
        score, ctx = score_candidate(c, sem_map.get(cid))
        verdict, concerns = classify(c, ctx)
        f = ctx["features"]
        p = c.get("profile", {})
        rows.append(
            {
                "rank": rank,
                "candidate_id": cid,
                "submission_score": sub_score,
                "raw_score": round(score, 4),
                "verdict": verdict,
                "title": p.get("current_title", ""),
                "title_tier": title_tier(p.get("current_title", "")),
                "yoe": p.get("years_of_experience"),
                "location": p.get("location", ""),
                "country": p.get("country", ""),
                "semantic": round(ctx.get("semantic_score", 0), 3),
                "title_score": round(f["title_score"], 2),
                "career_score": round(f["career_score"], 2),
                "skill_score": round(f["skill_score"], 2),
                "experience_score": round(f["experience_score"], 2),
                "behavioral_mult": round(ctx.get("behavioral_mult", 1), 2),
                "concerns": concerns,
                "matched_skills": f.get("matched_skills", [])[:5],
            }
        )

    counts = Counter(r["verdict"] for r in rows)
    tier_counts = Counter(r["title_tier"] for r in rows)
    honeypots = sum(1 for r in rows if r["verdict"] == "REJECT")
    with_concerns = sum(1 for r in rows if r["concerns"])

    print("=== TOP-100 VERIFICATION ===")
    print(f"Submission: {args.submission}")
    print(f"Verdicts: {dict(counts)}")
    print(f"Title tiers: {dict(tier_counts)}")
    print(f"With concerns (non-fatal): {with_concerns}")
    print(f"Semantic range: {min(r['semantic'] for r in rows):.3f} - {max(r['semantic'] for r in rows):.3f}")
    print(f"YoE range: {min(r['yoe'] for r in rows)} - {max(r['yoe'] for r in rows)}")
    print()

    for r in rows:
        flag = f" [{', '.join(r['concerns'])}]" if r["concerns"] else ""
        print(
            f"#{r['rank']:3d} {r['verdict']:6s} {r['candidate_id']} "
            f"sem={r['semantic']:.2f} {r['title'][:42]:42s} yoe={r['yoe']}{flag}"
        )

    if args.time_rank:
        print("\n=== RANK TIMING (3 runs, CPU) ===")
        times: list[float] = []
        for i in range(3):
            t0 = time.perf_counter()
            rank_candidates(args.candidates, top_k=100, artifacts_dir=args.artifacts)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  run {i+1}: {elapsed:.2f}s")
        print(f"  mean: {sum(times)/len(times):.2f}s  min: {min(times):.2f}s  max: {max(times):.2f}s")

    args.json_out.write_text(json.dumps({"summary": dict(counts), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nReport saved: {args.json_out}")
    return 0 if counts.get("REJECT", 0) == 0 and counts.get("WEAK", 0) <= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
