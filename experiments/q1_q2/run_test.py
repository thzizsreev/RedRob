#!/usr/bin/env python3
"""Build one Q1 and one Q2 vector; score five synthetic test cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from acceptance import evaluate_config_pass  # noqa: E402
from encode import build_q1_vector, build_q2_vector, load_vector_config  # noqa: E402
from paths import DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_SYNTHETIC  # noqa: E402
from report import write_acceptance_report, write_synthetic_results_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score TC1-TC5 with one Q1/Q2 facet centroid.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.synthetic, encoding="utf-8") as f:
        synthetic_cases = yaml.safe_load(f).get("cases", [])

    vector_config = load_vector_config(args.config)

    from tracks.instructor.core.encode import encode_candidates  # noqa: WPS433
    from experiments.stage3.shared.cpu_embedder import load_embedder  # noqa: WPS433

    print("Loading INSTRUCTOR ONNX embedder...")
    model = load_embedder()

    print("Building Q1 vector (facet centroid)...")
    q1_vec = build_q1_vector(model, vector_config)
    print("Building Q2 vector (facet centroid)...")
    q2_vec = build_q2_vector(model, vector_config)

    print(f"Encoding {len(synthetic_cases)} synthetic cases...")
    case_scores: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for case in synthetic_cases:
        case_id = case["id"]
        vec = encode_candidates(model, [case["text"].strip()], batch_size=1)[0]
        q1_score = float(np.dot(vec, q1_vec))
        q2_score = float(np.dot(vec, q2_vec))
        case_scores[case_id] = {"q1": q1_score, "q2": q2_score}
        rows.append(
            {
                "case_id": case_id,
                "case_name": case.get("name", case_id),
                "q1_score": q1_score,
                "q2_score": q2_score,
            }
        )
        print(f"  {case_id}: Q1={q1_score:.4f}  Q2={q2_score:.4f}")

    passed, checks = evaluate_config_pass(case_scores=case_scores)
    print(f"\nPASS: {'YES' if passed else 'NO'}")
    for chk in checks:
        mark = "ok" if chk["passed"] else "FAIL"
        print(f"  [{mark}] {chk['rule']}: {chk['detail']}")

    write_synthetic_results_csv(args.output / "synthetic_test_results.csv", rows)
    write_acceptance_report(
        args.output / "acceptance_report.json",
        {"passed": passed, "checks": checks, "scores": case_scores},
    )
    print(f"\nWrote {args.output / 'acceptance_report.json'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
