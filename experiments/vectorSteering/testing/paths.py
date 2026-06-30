"""Path constants for the vector steering feasibility test harness."""

from __future__ import annotations

from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TEST_DIR / "input" / "config.yaml"
DEFAULT_OUTPUT = TEST_DIR / "output"

PHASE_A_CSV = "phase_a_results.csv"
PHASE_B_CSV = "phase_b_results.csv"
SUMMARY_JSON = "summary.json"
