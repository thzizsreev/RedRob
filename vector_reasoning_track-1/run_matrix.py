#!/usr/bin/env python3
"""Run all 9 experiment-matrix cases → results/matrix.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import EXPERIMENT_MATRIX, RESUME_SECTION_TEXT
from run_experiment import run_pipeline
from validate_decode import validate_roundtrip

RESULTS_DIR = _ROOT / "results"


def main() -> None:
    passed, total = validate_roundtrip()
    if passed < total:
        print(
            f"WARNING: LangVAE gate failed ({passed}/{total}). "
            "Matrix outputs may be unrelated.",
            file=sys.stderr,
        )

    results = []
    for case in EXPERIMENT_MATRIX:
        print(f"Running test {case['id']}...")
        output = run_pipeline(
            resume_text=RESUME_SECTION_TEXT,
            s_tech=case["s_tech"],
            s_career=case["s_career"],
            s_behav=case["s_behav"],
            beta=case["beta"],
        )
        results.append(
            {
                "test_id": case["id"],
                "expected_direction": case["expected_direction"],
                "scores": output["scores"],
                "beta": output["beta"],
                "clauses": output["clauses"],
                "reasoning": output["reasoning"],
            }
        )
        print(f"  -> {output['reasoning'][:120]}...")
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "matrix.json"
    payload = {
        "decode_gate_passed": passed == total,
        "resume_section_text": RESUME_SECTION_TEXT,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(results)} test results to {output_path}")


if __name__ == "__main__":
    main()
