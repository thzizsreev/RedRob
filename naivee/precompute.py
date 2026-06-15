#!/usr/bin/env python3
"""Offline: vectorize sample candidates into a naive FAISS resume index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python naivee/precompute.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ARTIFACTS_DIR, SAMPLE_CANDIDATES_PATH
from encode import encode_passages, load_encoder
from passage import build_passage
from store import save_index


def load_sample_candidates(path: Path = SAMPLE_CANDIDATES_PATH) -> list[dict]:
    print(f"[1/4] Loading candidates from {path}...")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    print(f"      Loaded {len(records)} candidate records")
    return records


def run_precompute() -> None:
    print("=" * 60)
    print("NAIVEE PRECOMPUTE — building resume vector index")
    print("=" * 60)

    records = load_sample_candidates()

    print(f"\n[2/4] Building passages (summary + career descriptions)...")
    passages: list[tuple[str, str]] = []
    for i, record in enumerate(records, start=1):
        candidate_id = record["candidate_id"]
        passage = build_passage(record)
        char_count = len(passage)
        role_count = len(record.get("career_history", []))
        passages.append((candidate_id, passage))
        print(
            f"      [{i:2d}/{len(records)}] {candidate_id}  "
            f"roles={role_count}  chars={char_count:,}"
        )

    print(f"\n[3/4] Encoding {len(passages)} passages with BGE...")
    model = load_encoder()
    vectors = encode_passages(model, [p for _, p in passages])
    print(f"      Encoded matrix shape: {vectors.shape}")

    print(f"\n[4/4] Saving FAISS index to {ARTIFACTS_DIR}...")
    save_index(vectors, passages, ARTIFACTS_DIR)

    print("\n" + "=" * 60)
    print("Naivee precompute complete.")
    print("=" * 60)


def main() -> None:
    run_precompute()


if __name__ == "__main__":
    main()
