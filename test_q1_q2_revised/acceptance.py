"""Pass/fail rules for the five synthetic test cases."""

from __future__ import annotations

from typing import Any


def evaluate_config_pass(
    *,
    case_scores: dict[str, dict[str, float]],
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Pass iff all of:
      TC1 Q1 >= 0.90
      TC2 Q1 >= 0.87 AND TC2 Q2 >= 0.86
      TC3 Q1 >= 0.88
      TC4 Q2 <= 0.78 AND (TC4 Q1 - TC4 Q2) >= 0.06
      TC5 Q1 <= 0.82 AND (TC1 Q1 - TC5 Q1) >= 0.08
    """
    checks: list[dict[str, Any]] = []

    def add(rule: str, passed: bool, detail: str) -> None:
        checks.append({"rule": rule, "passed": passed, "detail": detail})

    tc1_q1 = case_scores.get("TC1", {}).get("q1")
    tc2_q1 = case_scores.get("TC2", {}).get("q1")
    tc2_q2 = case_scores.get("TC2", {}).get("q2")
    tc3_q1 = case_scores.get("TC3", {}).get("q1")
    tc4_q1 = case_scores.get("TC4", {}).get("q1")
    tc4_q2 = case_scores.get("TC4", {}).get("q2")
    tc5_q1 = case_scores.get("TC5", {}).get("q1")

    if tc1_q1 is not None:
        add("TC1_Q1", tc1_q1 >= 0.90, f"q1={tc1_q1:.4f}, need >= 0.90")
    if tc2_q1 is not None:
        add("TC2_Q1", tc2_q1 >= 0.87, f"q1={tc2_q1:.4f}, need >= 0.87")
    if tc2_q2 is not None:
        add("TC2_Q2", tc2_q2 >= 0.86, f"q2={tc2_q2:.4f}, need >= 0.86")
    if tc3_q1 is not None:
        add("TC3_Q1", tc3_q1 >= 0.88, f"q1={tc3_q1:.4f}, need >= 0.88")
    if tc4_q2 is not None:
        add("TC4_Q2", tc4_q2 <= 0.78, f"q2={tc4_q2:.4f}, need <= 0.78")
    if tc4_q1 is not None and tc4_q2 is not None:
        gap = tc4_q1 - tc4_q2
        add("TC4_Q1_minus_Q2", gap >= 0.06, f"gap={gap:.4f}, need >= 0.06")
    if tc5_q1 is not None:
        add("TC5_Q1", tc5_q1 <= 0.82, f"q1={tc5_q1:.4f}, need <= 0.82")
    if tc1_q1 is not None and tc5_q1 is not None:
        gap = tc1_q1 - tc5_q1
        add("TC5_TC1_minus_Q1", gap >= 0.08, f"gap={gap:.4f}, need >= 0.08")

    return all(c["passed"] for c in checks), checks
