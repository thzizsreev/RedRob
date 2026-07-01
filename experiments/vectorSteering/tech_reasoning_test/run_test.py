#!/usr/bin/env python3
"""Run full 3-sentence reasoning builder on test candidates."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parents[3]
INPUT_PATH = PACKAGE_DIR / "input" / "candidates.json"
OUTPUT_PATH = PACKAGE_DIR / "output" / "reasoning_output.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from reasoning_builder import build_reasoning

MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"

MIRA_ID = "CAND_0018549"

MIRA_RAW_CHECKS: list[tuple[str, str, str]] = [
    ("S1 starts with name", "startswith", "Mira Verma brings"),
    ("S1 contains companies", "contains", "at Uber and Flipkart"),
    ("S1 contains system", "contains", "ranking pipeline"),
    ("S1 contains metric", "contains", "improving revenue-per-search by 12%"),
    ("S1 contains skill", "contains", "Elasticsearch"),
    ("S1 ends with alignment", "endswith", "retrieval and ranking requirements"),
    ("S2 career type", "contains_ci", "career is entirely at product companies"),
    ("S2 tenure", "contains", "stable tenure"),
    ("S2 pre-LLM", "contains", "pre-LLM production ML ownership"),
    ("S2 disqualifiers", "contains", "clearing the JD's explicit disqualifiers"),
    ("S3 activity", "contains", "63 days since last platform login"),
    ("S3 notice", "contains", "60-day notice period"),
    ("S3 friction", "contains", "moderate friction"),
    ("S3 outreach", "endswith", "timeline negotiation"),
]


def _check_value(actual: str, check_type: str, expected: str) -> bool:
    if check_type == "startswith":
        return actual.startswith(expected)
    if check_type == "endswith":
        return actual.endswith(expected)
    if check_type == "contains":
        return expected in actual
    if check_type == "contains_ci":
        return expected.lower() in actual.lower()
    raise ValueError(f"Unknown check type: {check_type}")


def validate_mira_raw(result: dict) -> bool:
    print(f"\n=== Mira raw validation ({MIRA_ID}) ===")
    all_pass = True
    for label, check_type, expected in MIRA_RAW_CHECKS:
        if label.startswith("S1"):
            actual = result["s1_raw"]
        elif label.startswith("S2"):
            actual = result["s2_raw"]
        else:
            actual = result["s3_raw"]

        passed = _check_value(actual, check_type, expected)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            print(f"         expected: {expected!r}")
            print(f"         actual:   {actual!r}")
            all_pass = False
    return all_pass


def _print_candidate(result: dict) -> None:
    cid = result["candidate_id"]
    tech_cat = result["tech_cat"]
    print(f"\n=== {cid} ({tech_cat}) ===")
    print(f"S1 RAW:         {result['s1_raw']}")
    print(f"S2 RAW:         {result['s2_raw']}")
    print(f"S3 RAW:         {result['s3_raw']}")
    print(f"S1 PARAPHRASED: {result['s1_paraphrased']}")
    print(f"S2 PARAPHRASED: {result['s2_paraphrased']}")
    print(f"S3 PARAPHRASED: {result['s3_paraphrased']}")
    print(f"REASONING:      {result['reasoning']}")


def run(input_path: Path, output_path: Path) -> list[dict]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "candidates" in payload:
        candidates = payload["candidates"]
    elif isinstance(payload, list):
        candidates = payload
    else:
        raise ValueError('Input must be a list of candidates or {"candidates": [...]}')

    def _stub_paraphrase(text: str, _temperature: float) -> str:
        return text

    print("Phase 1 — raw sentence validation (no model)...")
    mira_result: dict | None = None
    for candidate in candidates:
        result = build_reasoning(candidate, _stub_paraphrase)
        if result["candidate_id"] == MIRA_ID:
            mira_result = result

    if mira_result is None:
        print(f"ERROR: {MIRA_ID} not found in input.")
        sys.exit(1)

    if not validate_mira_raw(mira_result):
        print("\nStopping: Mira raw validation failed. Paraphraser not loaded.")
        sys.exit(1)

    print("\nAll Mira raw checks passed. Phase 2 — loading paraphraser...")
    from paraphrase import load_paraphraser

    paraphrase_fn = load_paraphraser()
    print("Model loaded.\n")

    results: list[dict] = []
    for candidate in candidates:
        result = build_reasoning(candidate, paraphrase_fn)
        _print_candidate(result)
        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "meta": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "candidate_count": len(results),
                    "model": MODEL_NAME,
                    "pipeline_version": "stage6_reasoning_builder",
                    "variation_mode": "cpu_onnx_encoder_seeded_temperature_once_per_slot",
                    "input_path": str(input_path),
                    "mira_raw_validation": "PASS",
                },
                "candidates": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved results -> {output_path}")
    return results


def main() -> int:
    try:
        run(INPUT_PATH, OUTPUT_PATH)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
