"""Shared utilities for text cleaning, config loading, and safe getters."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = ROOT_DIR / "config"

_PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.25,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Path | None = None) -> tuple[dict, dict]:
    base = config_dir or DEFAULT_CONFIG_DIR
    jd_terms = load_yaml(base / "jd_terms.yaml")
    weights = load_yaml(base / "scoring_weights.yaml")
    return jd_terms, weights


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def proficiency_weight(proficiency: str) -> float:
    return _PROFICIENCY_WEIGHTS.get(str(proficiency).lower(), 0.3)


def contains_term(text: str, term: str) -> bool:
    """Word-boundary aware substring match for multi-word terms."""
    term = term.lower().strip()
    if not term or not text:
        return False
    pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def find_matching_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if contains_term(text, t)]


def title_matches(title: str, title_list: list[str]) -> bool:
    title_lower = clean_text(title)
    for pattern in title_list:
        if contains_term(title_lower, pattern.lower()):
            return True
    return False
