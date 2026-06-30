"""Structured output schema for honeypot LLM judgments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VERDICTS = frozenset({"honeypot", "not_honeypot", "uncertain"})
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
TITLE_DESC_CONSISTENCY = frozenset({"yes", "no", "partial"})
CONTRADICTION_TYPES = frozenset(
    {
        "tenure_founding_mismatch",
        "proficiency_duration_mismatch",
        "timeline_arithmetic",
        "skill_breadth_implausibility",
        "narrative_incoherence",
        "title_description_mismatch",
        "skills_array_unsupported",
        "synthetic_pattern_no_specific_contradiction",
        "other",
        "none",
    }
)


@dataclass
class CitedField:
    field: str
    value: str
    relevance: str

    @classmethod
    def from_dict(cls, data: dict) -> CitedField:
        return cls(
            field=str(data.get("field", "")),
            value=str(data.get("value", "")),
            relevance=str(data.get("relevance", "")),
        )


@dataclass
class HoneypotJudgment:
    candidate_id: str
    verdict: str
    confidence: str
    contradiction_type: str
    cited_fields: list[CitedField]
    title_description_consistent: str
    reasoning: str

    @classmethod
    def from_dict(cls, data: dict, *, expected_id: str | None = None) -> HoneypotJudgment:
        candidate_id = str(data.get("candidate_id", ""))
        if expected_id and candidate_id != expected_id:
            raise ValueError(
                f"candidate_id mismatch: expected {expected_id!r}, got {candidate_id!r}"
            )

        verdict = str(data.get("verdict", "")).strip().lower()
        confidence = str(data.get("confidence", "")).strip().lower()
        contradiction_type = str(data.get("contradiction_type", "")).strip().lower()
        title_description_consistent = (
            str(data.get("title_description_consistent", "")).strip().lower()
        )

        if verdict not in VERDICTS:
            raise ValueError(f"invalid verdict: {verdict!r}")
        if confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"invalid confidence: {confidence!r}")
        if title_description_consistent not in TITLE_DESC_CONSISTENCY:
            raise ValueError(
                f"invalid title_description_consistent: {title_description_consistent!r}"
            )
        if contradiction_type not in CONTRADICTION_TYPES:
            raise ValueError(f"invalid contradiction_type: {contradiction_type!r}")

        raw_cited = data.get("cited_fields")
        if not isinstance(raw_cited, list):
            raise ValueError("cited_fields must be a list")

        cited_fields = [CitedField.from_dict(item) for item in raw_cited]
        reasoning = str(data.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("reasoning must be non-empty")

        return cls(
            candidate_id=candidate_id,
            verdict=verdict,
            confidence=confidence,
            contradiction_type=contradiction_type,
            cited_fields=cited_fields,
            title_description_consistent=title_description_consistent,
            reasoning=reasoning,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cited_fields"] = [asdict(cf) for cf in self.cited_fields]
        return d


def needs_pass2(judgment: HoneypotJudgment) -> bool:
    return judgment.verdict == "uncertain" or (
        judgment.verdict == "honeypot" and judgment.confidence == "low"
    )


def parse_judgment_json(raw: dict, *, expected_id: str) -> HoneypotJudgment:
    return HoneypotJudgment.from_dict(raw, expected_id=expected_id)
