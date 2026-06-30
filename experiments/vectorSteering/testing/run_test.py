#!/usr/bin/env python3
"""Orchestrator: Phase A -> gate -> Phase B -> summary."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

# --- paths (edit here) ---
INPUT_CONFIG = TEST_DIR / "input" / "config.yaml"
OUTPUT_DIR = TEST_DIR / "output"

# --- run settings (edit here) ---
RUN_PHASE = "all"  # "a", "b", or "all"
FORCE_PHASE_B = False

SUMMARY_JSON = "summary.json"
PHASE_A_CSV = "phase_a_results.csv"
PHASE_B_CSV = "phase_b_results.csv"


def _write_error_summary(output_dir: Path, error_message: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON
    payload = {
        "phase_a_ran": False,
        "phase_b_ran": False,
        "phase_b_skipped_reason": None,
        "error": error_message,
        "overall_automated_pass": False,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return summary_path


def main() -> int:
    print(f"Input config: {INPUT_CONFIG}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"Run phase:    {RUN_PHASE}")
    print(flush=True)

    if not INPUT_CONFIG.is_file():
        print(f"ERROR: config not found: {INPUT_CONFIG}")
        summary_path = _write_error_summary(OUTPUT_DIR, f"config not found: {INPUT_CONFIG}")
        print(f"Wrote error summary -> {summary_path}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from acceptance import build_summary, print_review_checklist  # noqa: WPS433
    from report import write_summary_json  # noqa: WPS433
    from run_phase_a import run_phase_a  # noqa: WPS433
    from run_phase_b import phase_a_passed, run_phase_b  # noqa: WPS433

    phase_a_rows = None
    phase_b_rows = None
    phase_a_ran = False
    phase_b_ran = False
    phase_b_skipped_reason = None
    error_message = None

    try:
        if RUN_PHASE in ("a", "all"):
            print("=== Phase A: Baseline inversion stability ===\n", flush=True)
            phase_a_rows = run_phase_a(INPUT_CONFIG, OUTPUT_DIR)
            phase_a_ran = True

        if RUN_PHASE in ("b", "all"):
            if RUN_PHASE == "b" or FORCE_PHASE_B:
                gate_ok = True
                skip_reason = None
            else:
                gate_ok, skip_reason = phase_a_passed(OUTPUT_DIR)

            if gate_ok or FORCE_PHASE_B:
                if not gate_ok:
                    print(
                        f"Warning: Phase A gate failed ({skip_reason}); "
                        "running Phase B anyway (FORCE_PHASE_B=True).",
                        flush=True,
                    )
                print("\n=== Phase B: Steering direction sanity check ===\n", flush=True)
                phase_b_rows = run_phase_b(INPUT_CONFIG, OUTPUT_DIR)
                phase_b_ran = True
            else:
                phase_b_skipped_reason = skip_reason
                print(f"\nPhase B skipped: {skip_reason}", flush=True)
                print("Set FORCE_PHASE_B = True in run_test.py to override.", flush=True)

    except Exception as exc:
        error_message = str(exc)
        print(f"\nERROR during test run: {exc}", flush=True)
        traceback.print_exc()

    summary = build_summary(
        phase_a_rows=phase_a_rows,
        phase_b_rows=phase_b_rows,
        phase_a_ran=phase_a_ran,
        phase_b_ran=phase_b_ran,
        phase_b_skipped_reason=phase_b_skipped_reason,
    )
    if error_message:
        summary["error"] = error_message

    summary_path = OUTPUT_DIR / SUMMARY_JSON
    write_summary_json(summary_path, summary)

    print(f"\nWrote summary -> {summary_path}", flush=True)
    if phase_a_ran:
        print(f"  Phase A CSV -> {OUTPUT_DIR / PHASE_A_CSV}", flush=True)
    if phase_b_ran:
        print(f"  Phase B CSV -> {OUTPUT_DIR / PHASE_B_CSV}", flush=True)

    if error_message:
        return 1

    print(f"Overall automated pass: {'YES' if summary['overall_automated_pass'] else 'NO'}", flush=True)

    if phase_a_ran and summary.get("automated", {}).get("phase_a"):
        det = summary["automated"]["phase_a"]["determinism"]
        print(f"  Phase A determinism: {det['detail']}", flush=True)

    if phase_b_ran and summary.get("automated", {}).get("phase_b"):
        det = summary["automated"]["phase_b"]["determinism"]
        print(f"  Phase B determinism: {det['detail']}", flush=True)

    print_review_checklist()

    return 0 if summary["overall_automated_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
