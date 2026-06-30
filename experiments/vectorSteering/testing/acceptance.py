"""Automated determinism checks and pass/fail scaffolding."""

from __future__ import annotations

import re
from typing import Any


def normalize_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def texts_near_identical(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


def group_deterministic(texts: list[str]) -> bool:
    if not texts:
        return False
    first = normalize_text(texts[0])
    return all(normalize_text(t) == first for t in texts)


def evaluate_phase_a_determinism(rows: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """
    Pass if at least 4 of 5 sentences have stable decodes across all repeats.
    """
    by_id: dict[str, list[str]] = {}
    for row in rows:
        by_id.setdefault(row["input_id"], []).append(row["decoded_text"])

    checks: list[dict[str, Any]] = []
    stable_count = 0
    for input_id, decoded_texts in sorted(by_id.items()):
        stable = group_deterministic(decoded_texts)
        if stable:
            stable_count += 1
        checks.append(
            {
                "input_id": input_id,
                "rule": "determinism",
                "passed": stable,
                "detail": f"{len(decoded_texts)} decodes, stable={stable}",
            }
        )

    threshold = 4
    passed = stable_count >= threshold
    summary = {
        "criterion": "determinism",
        "passed": passed,
        "stable_sentences": stable_count,
        "total_sentences": len(by_id),
        "threshold": threshold,
        "detail": f"{stable_count}/{len(by_id)} sentences stable (need >= {threshold})",
    }
    return passed, checks, summary


def evaluate_phase_b_determinism(rows: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """
    Pass if at least 5 of 7 S values are stable across all sweep repeats.
    """
    by_s: dict[str, list[str]] = {}
    for row in rows:
        key = str(row["s_value"])
        by_s.setdefault(key, []).append(row["decoded_text"])

    checks: list[dict[str, Any]] = []
    stable_count = 0
    for s_key, decoded_texts in sorted(by_s.items(), key=lambda kv: float(kv[0])):
        stable = group_deterministic(decoded_texts)
        if stable:
            stable_count += 1
        checks.append(
            {
                "s_value": float(s_key),
                "rule": "determinism",
                "passed": stable,
                "detail": f"{len(decoded_texts)} sweeps, stable={stable}",
            }
        )

    threshold = 5
    passed = stable_count >= threshold
    summary = {
        "criterion": "determinism",
        "passed": passed,
        "stable_s_values": stable_count,
        "total_s_values": len(by_s),
        "threshold": threshold,
        "detail": f"{stable_count}/{len(by_s)} S values stable (need >= {threshold})",
    }
    return passed, checks, summary


def build_summary(
    *,
    phase_a_rows: list[dict[str, Any]] | None,
    phase_b_rows: list[dict[str, Any]] | None,
    phase_a_ran: bool,
    phase_b_ran: bool,
    phase_b_skipped_reason: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "phase_a_ran": phase_a_ran,
        "phase_b_ran": phase_b_ran,
        "phase_b_skipped_reason": phase_b_skipped_reason,
        "automated": {},
        "human_review": {
            "phase_a_fidelity": "pending_human_review",
            "phase_b_monotonicity": "pending_human_review",
            "phase_b_coherence": "pending_human_review",
            "phase_b_overshoot": "pending_human_review",
        },
        "overall_automated_pass": False,
    }

    automated_pass = True

    if phase_a_rows:
        a_pass, a_checks, a_summary = evaluate_phase_a_determinism(phase_a_rows)
        summary["automated"]["phase_a"] = {
            "determinism": a_summary,
            "checks": a_checks,
        }
        automated_pass = automated_pass and a_pass

    if phase_b_rows:
        b_pass, b_checks, b_summary = evaluate_phase_b_determinism(phase_b_rows)
        summary["automated"]["phase_b"] = {
            "determinism": b_summary,
            "checks": b_checks,
        }
        automated_pass = automated_pass and b_pass

    summary["overall_automated_pass"] = automated_pass
    return summary


def print_review_checklist() -> None:
    print("\n--- Human review checklist (fill CSV columns, then re-evaluate) ---")
    print("Phase A: human_fidelity — does decoded text preserve meaning? (need 4/5)")
    print("Phase B: human_coherent — grammatically valid at each S? (need 6/7)")
    print("Phase B: human_sentiment — negative-to-positive trend across S?")
    print("Phase B: human_sentiment at S=-0.25 and S=1.25 — overshoot still coherent?")
    print("See ../vector_steering_test_plan.md Section 5 for full pass/fail thresholds.")
