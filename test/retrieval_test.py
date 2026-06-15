#!/usr/bin/env python3
"""
Simple retrieval test — search the index with a query string.

Run from project root (after precompute.py):
    python test/retrieval_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import SAMPLE_CANDIDATES_PATH
from rank import retrieve_from_text

# --- edit these ---
QUERY = "production embedding retrieval FAISS vector database ranking"
TOP_K = 10
# ------------------


def load_candidate_lookup(path: Path = SAMPLE_CANDIDATES_PATH) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {record["candidate_id"]: record for record in records}


def main() -> None:
    candidates = load_candidate_lookup()
    results = retrieve_from_text(QUERY, k=TOP_K)

    print(f'Query: "{QUERY}"\n')
    print(f"{'Rank':<5} {'Score':>8}  {'Candidate ID':<14}  Title")
    print("-" * 70)

    for hit in results:
        profile = candidates.get(hit.candidate_id, {}).get("profile", {})
        title = profile.get("current_title", "(unknown)")
        print(f"{hit.rank:<5} {hit.score:8.4f}  {hit.candidate_id:<14}  {title}")

    print(f"\n{len(results)} results returned.")


if __name__ == "__main__":
    main()
