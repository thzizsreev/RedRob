"""Post-rank safety checks for submission top-K."""

from __future__ import annotations

from ranker import jd_config as jd
from ranker.honeypot import is_honeypot


def is_submission_safe(candidate: dict, context: dict) -> tuple[bool, str]:
    """Return (safe, reason) for inclusion in final top-100."""
    hp, hp_flags = is_honeypot(candidate)
    if hp:
        return False, f"hard_honeypot:{','.join(hp_flags)}"

    title = candidate.get("profile", {}).get("current_title", "")
    if jd.TITLE_NEGATIVE.search(title):
        return False, "negative_title"

    f = context.get("features", {})
    sem = context.get("semantic_score", 0.0) or 0.0
    trap_risk = context.get("trap_risk", 0.0) or 0.0

    if trap_risk >= 0.35:
        return False, f"high_trap_risk:{trap_risk:.2f}"

    if f.get("title_score", 0) < 0.25 and sem < 0.55:
        return False, "weak_title_low_semantic"

    return True, "ok"
