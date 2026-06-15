#!/usr/bin/env python3
"""Online: rank sample candidates by JD query against the naive FAISS index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ARTIFACTS_DIR, JD_QUERY_TEXT, RANK_RESULTS_FILENAME
from encode import encode_query, load_encoder
from store import load_index


def run_rank(*, k: int = 50) -> list[dict]:
    print("=" * 60)
    print("NAIVEE RANK — JD query retrieval")
    print("=" * 60)

    print(f"\n[1/4] Loading encoder...")
    model = load_encoder()

    print(f"\n[2/4] Loading FAISS index from {ARTIFACTS_DIR}...")
    index, id_map = load_index(ARTIFACTS_DIR)

    print(f"\n[3/4] Encoding JD query ({len(JD_QUERY_TEXT)} chars)...")
    print(f"      Query preview: {JD_QUERY_TEXT[:80]}...")
    query_vec = encode_query(model, JD_QUERY_TEXT)
    print(f"      Query vector shape: {query_vec.shape}")

    k_actual = min(k, index.ntotal)
    print(f"\n[4/4] Searching top-{k_actual} from {index.ntotal} indexed candidates...")
    scores, indices = index.search(query_vec.reshape(1, -1), k_actual)

    results: list[dict] = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        results.append(
            {
                "rank": rank,
                "candidate_id": id_map[int(idx)],
                "score": float(score),
            }
        )

    print(f"      Retrieved {len(results)} candidates")
    return results


def main() -> None:
    results = run_rank(k=50)

    print(f"\nTop {min(10, len(results))} candidates:\n")
    for hit in results[:10]:
        print(f"  {hit['rank']:3d}. {hit['candidate_id']}  score={hit['score']:.4f}")

    output_path = ARTIFACTS_DIR / RANK_RESULTS_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} results to {output_path}")
    print("=" * 60)
    print("Naivee rank complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
