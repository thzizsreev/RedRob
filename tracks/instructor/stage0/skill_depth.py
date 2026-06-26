"""Per-skill depth scoring from proficiency and duration."""

from __future__ import annotations

import math

PROFICIENCY_SCORES = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}


def depth_score(proficiency: str | None, duration_months: int | None) -> float:
    prof = PROFICIENCY_SCORES.get(str(proficiency or "").lower(), 0.5)
    years = max(float(duration_months or 0) / 12.0, 0.0)
    duration_part = min(math.log(1.0 + years) / math.log(11.0), 1.0)
    return 0.6 * duration_part + 0.4 * prof
