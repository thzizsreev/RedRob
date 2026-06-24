"""Jargon text extraction for BM25 sparse retrieval."""

from __future__ import annotations

import re

_EMPTY_JARGON = "no skills listed"


def _skill_name(skill: dict) -> str:
    return str(skill.get("name") or skill.get("skill_name") or "").strip()


def _description_jargon_tokens(description: str, q4_tokens: frozenset[str] | None) -> list[str]:
    if not q4_tokens or not description:
        return []
    desc_lower = description.lower()
    found: list[str] = []
    for token in q4_tokens:
        token_lower = token.lower()
        if " " in token_lower:
            if token_lower in desc_lower:
                found.append(token_lower)
        elif re.search(rf"\b{re.escape(token_lower)}\b", desc_lower):
            found.append(token_lower)
    return found


def build_jargon_text(
    record: dict,
    *,
    q4_tokens: frozenset[str] | None = None,
) -> str:
    """Build a clean jargon block for BM25 indexing."""
    parts: list[str] = []

    for skill in record.get("skills", []):
        name = _skill_name(skill)
        if name:
            parts.append(name)

    profile = record.get("profile", {})
    current_title = profile.get("current_title", record.get("current_title", "")).strip()
    if current_title:
        parts.append(current_title)

    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)

    if q4_tokens:
        seen_desc_tokens: set[str] = set()
        for exp in record.get("career_history", record.get("experience", [])):
            description = exp.get("description", "").strip()
            for token in _description_jargon_tokens(description, q4_tokens):
                if token not in seen_desc_tokens:
                    seen_desc_tokens.add(token)
                    parts.append(token)

    text = " ".join(parts).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else _EMPTY_JARGON


def tokenize_jargon(text: str) -> list[str]:
    """Simple whitespace tokenization for rank_bm25."""
    return text.lower().split()
