#!/usr/bin/env python3
"""
Rule-based candidate ranking for RedRob hackathon.

Loads candidates, scores across technical/career/behavioral/logistics dimensions,
applies risk penalties, and writes a top-100 submission CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.data_loader import load_candidates
from src.feature_extraction import extract_features
from src.normalizer import normalize_candidate
from src.output_writer import RankedCandidate, write_submission_csv
from src.reasoning import generate_reasoning
from src.risk_detection import compute_risk_penalty
from src.scoring import compute_final_score
from src.utils import DEFAULT_CONFIG_DIR, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidates for Senior AI Engineer role (rule-based, offline)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/candidates.jsonl.gz"),
        help="Path to candidates.jsonl, .jsonl.gz, or .json array",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing jd_terms.yaml and scoring_weights.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N candidates (for testing)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of candidates to include in output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    jd_terms, weights = load_config(args.config_dir)
    ranked: list[RankedCandidate] = []
    processed = 0

    print(f"Loading candidates from {args.input}...")
    for raw in load_candidates(args.input, limit=args.limit):
        candidate = normalize_candidate(raw)
        features = extract_features(candidate, jd_terms)
        risk_penalty, risk_flags = compute_risk_penalty(
            features, candidate, weights
        )
        features.risk_flags = risk_flags

        scores = compute_final_score(
            features, candidate, jd_terms, weights, risk_penalty
        )
        reasoning = generate_reasoning(candidate, features, scores)

        ranked.append(
            RankedCandidate(
                candidate_id=candidate.candidate_id,
                score=scores.final_score,
                reasoning=reasoning,
            )
        )
        processed += 1
        if processed % 10000 == 0:
            print(f"  scored {processed:,} candidates...")

    if not ranked:
        print("Error: no candidates processed.", file=sys.stderr)
        sys.exit(1)

    top_n = min(args.top_n, len(ranked))
    if top_n < args.top_n:
        print(
            f"Warning: only {len(ranked)} candidates available; "
            f"outputting top {top_n}.",
            file=sys.stderr,
        )

    write_submission_csv(ranked, args.output, top_n=top_n)

    elapsed = time.perf_counter() - start
    print(f"Ranked {processed:,} candidates in {elapsed:.1f}s")
    print(f"Wrote top {top_n} to {args.output}")


if __name__ == "__main__":
    main()
