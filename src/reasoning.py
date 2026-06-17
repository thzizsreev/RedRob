"""Generate candidate-specific reasoning strings from extracted facts."""

from __future__ import annotations

from src.feature_extraction import CandidateFeatures
from src.normalizer import CandidateRecord
from src.scoring import ScoreBreakdown


def generate_reasoning(
    candidate: CandidateRecord,
    features: CandidateFeatures,
    scores: ScoreBreakdown,
) -> str:
    parts: list[str] = []

    years = features.years_of_experience
    title = features.current_title or "professional"

    if features.top_career_terms:
        terms = ", ".join(features.top_career_terms[:2])
        parts.append(
            f"Strong fit for Senior AI Engineer: career evidence in {terms}"
        )
    elif features.top_skill_names:
        skills = ", ".join(features.top_skill_names[:2])
        parts.append(
            f"Relevant technical skills ({skills}) align with retrieval/ranking JD"
        )
    else:
        parts.append(
            f"Moderate alignment as {title} with {years:.1f} years experience"
        )

    if features.has_production_language:
        prod = features.production_phrase_hits[0] if features.production_phrase_hits else "production"
        parts.append(f"shows {prod} deployment experience")
    elif features.hands_on_hits:
        parts.append("demonstrates hands-on engineering in career history")

    signals = candidate.redrob_signals
    behavioral_notes: list[str] = []
    if signals.open_to_work_flag:
        behavioral_notes.append("open to work")
    if signals.recruiter_response_rate >= 0.5:
        behavioral_notes.append("responsive to recruiters")
    if signals.willing_to_relocate:
        behavioral_notes.append("willing to relocate")
    if behavioral_notes:
        parts.append("; ".join(behavioral_notes).capitalize() + ".")

    if features.risk_flags:
        concern = features.risk_flags[0].replace("_", " ")
        parts.append(f"Note: {concern}.")

    if features.is_disqualifier_title and scores.technical < 40:
        parts.append(
            f"Current title ({title}) is non-engineering; ranked lower despite skill keywords."
        )

    text = " ".join(parts)
    if len(text) > 500:
        text = text[:497] + "..."
    return text
