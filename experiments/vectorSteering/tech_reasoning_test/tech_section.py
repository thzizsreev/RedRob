"""Backward-compatible re-exports — logic lives in tracks.instructor.stage6."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage6.reasoning_builder import (
    assemble_sentence_1,
    build_reasoning,
    calculate_tech_cat,
    extract_named_tech,
    extract_primary_metric,
    extract_verified_skill,
    stable_seed,
)

__all__ = [
    "assemble_sentence_1",
    "build_reasoning",
    "calculate_tech_cat",
    "extract_named_tech",
    "extract_primary_metric",
    "extract_verified_skill",
    "stable_seed",
]
