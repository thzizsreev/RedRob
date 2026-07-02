#!/usr/bin/env python3
"""Rank candidates with TRACER — Trap-aware Retrieval And Career Evidence Ranker."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent

from ranker.io import rank_candidates, write_submission


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRACER ranker — hybrid+MQ semantic, cross-encoder boost, trap-aware scoring."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl (or .jsonl.gz / sample .json array)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidates to rank (default: 100)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "artifacts",
        help="Directory with precomputed embeddings (from precompute_embeddings.py)",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"Error: candidates file not found: {args.candidates}", file=sys.stderr)
        return 1

    if not (args.artifacts / "meta.json").exists():
        print(
            "Warning: no embedding artifacts found — run scripts/precompute_embeddings.py first.",
            file=sys.stderr,
        )

    t0 = time.perf_counter()
    ranked = rank_candidates(args.candidates, top_k=args.top_k, artifacts_dir=args.artifacts)
    elapsed = time.perf_counter() - t0

    if len(ranked) < args.top_k:
        print(
            f"Warning: only {len(ranked)} candidates available (requested {args.top_k})",
            file=sys.stderr,
        )

    write_submission(ranked, args.out)
    print(f"[TRACER] Wrote {len(ranked)} rows to {args.out} in {elapsed:.2f}s")
    if ranked:
        ctx = ranked[0][3]
        sem = ctx.get("semantic_score", 0)
        print(
            f"Top candidate: {ranked[0][1]} "
            f"(raw={ranked[0][0]:.4f}, semantic={sem:.3f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
