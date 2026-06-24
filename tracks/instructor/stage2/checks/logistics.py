"""Feature F2 — location tier (no remove)."""

from __future__ import annotations

from dataclasses import dataclass

from tracks.instructor.stage2.checks._history import normalize_text
from tracks.instructor.stage2.config import Stage2Config


@dataclass(frozen=True)
class LogisticsResult:
    location_tier: str


def _location_text(record: dict) -> str:
    profile = record.get("profile") or {}
    parts = [profile.get("location"), profile.get("country")]
    return " ".join(normalize_text(str(p)) for p in parts if p)


def _matches_any(text: str, locations: list[str]) -> bool:
    return any(loc in text for loc in locations)


def evaluate_logistics(record: dict, config: Stage2Config) -> LogisticsResult:
    cfg = config.logistics
    text = _location_text(record)
    profile = record.get("profile") or {}
    country = normalize_text(str(profile.get("country", "")))

    tier = "unknown"
    if text:
        if _matches_any(text, cfg.preferred_locations):
            tier = "preferred"
        elif _matches_any(text, cfg.acceptable_locations):
            tier = "acceptable"
        elif country and not _matches_any(country, cfg.india_signals):
            tier = "outside_india"
        elif country and _matches_any(country, cfg.india_signals):
            tier = "acceptable"
        elif text and not _matches_any(text, cfg.india_signals + cfg.preferred_locations + cfg.acceptable_locations):
            if any(
                token in text
                for token in ("usa", "united states", "uk", "canada", "singapore", "dubai")
            ):
                tier = "outside_india"

    signals = record.get("redrob_signals") or {}
    willing = signals.get("willing_to_relocate")
    if willing is True and tier in ("acceptable", "unknown"):
        tier = "preferred" if tier == "acceptable" else "acceptable"

    return LogisticsResult(location_tier=tier)
