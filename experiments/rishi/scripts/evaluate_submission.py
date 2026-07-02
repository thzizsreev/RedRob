#!/usr/bin/env python3
"""Compare v1/v2/v3 submissions and print quality summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ranker.honeypot import is_honeypot
from ranker.scorer import score_candidate


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


def load_submission(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["rank"]), row["candidate_id"]))
    return sorted(rows, key=lambda x: x[0])


def summarize(name: str, path: Path, candidates: dict[str, dict], sem: dict[str, float] | None) -> None:
    if not path.exists():
        print(f"\n{name}: MISSING ({path})")
        return
    top = load_submission(path)[:10]
    hp_count = 0
    print(f"\n=== {name} ({path.name}) top-10 ===")
    for rank, cid in top:
        c = candidates[cid]
        p = c.get("profile", {})
        title = p.get("current_title", "")
        yoe = p.get("years_of_experience", "")
        hp, flags = is_honeypot(c)
        if hp:
            hp_count += 1
        s = sem.get(cid) if sem else None
        score, ctx = score_candidate(c, s)
        print(
            f"  #{rank} {cid} score={score:.3f} sem={ctx.get('semantic_score', 0):.3f} "
            f"title={title[:50]!r} yoe={yoe}"
            + (f" HONEYPOT={flags}" if hp else "")
        )

    all_top = load_submission(path)[:100]
    hp100 = sum(1 for _, cid in all_top if is_honeypot(candidates[cid])[0])
    print(f"  Honeypots in top-100: {hp100}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "candidates.jsonl")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    sem: dict[str, float] | None = None
    meta_path = args.artifacts / "meta.json"
    if meta_path.exists():
        import numpy as np

        ids = json.loads((args.artifacts / "candidate_ids.json").read_text())
        scores = np.load(args.artifacts / "semantic_scores.npy")
        sem = {cid: float(scores[i]) for i, cid in enumerate(ids)}
        meta = json.loads(meta_path.read_text())
        print(f"v3 artifacts: backend={meta.get('backend')} model={meta.get('model')}")

    base = PROJECT_ROOT
    summarize("v1", base / "working_v1" / "submission_v1.csv", candidates, None)
    summarize("v2", base / "working_v2" / "submission_v2.csv", candidates, sem)
    summarize("v3", ROOT / "submission_v3.csv", candidates, sem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
