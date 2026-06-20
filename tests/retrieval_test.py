#!/usr/bin/env python3
"""
Simple retrieval test — search the index with the precomputed JD query vector.

Run from project root (after precompute.py):
    python tests/retrieval_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rank import retrieve
from tracks.shared.paths import SAMPLE_CANDIDATES_PATH

TOP_K = 10


def load_candidate_lookup(path=SAMPLE_CANDIDATES_PATH) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {record["candidate_id"]: record for record in records}


def main() -> None:
    candidates = load_candidate_lookup()
    results = retrieve(k=TOP_K)

    print("Query: precomputed JD query vector (jd_query_vec.npy)\n")
    print(f"{'Rank':<5} {'Score':>8}  {'Candidate ID':<14}  Title")
    print("-" * 70)

    for hit in results:
        profile = candidates.get(hit.candidate_id, {}).get("profile", {})
        title = profile.get("current_title", "(unknown)")
        print(f"{hit.rank:<5} {hit.score:8.4f}  {hit.candidate_id:<14}  {title}")

    print(f"\n{len(results)} results returned.")


if __name__ == "__main__":
    main()
