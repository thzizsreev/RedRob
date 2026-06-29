"""Hardcoded anchors, resume, hyperparameters, and experiment matrix from plan_1_resume_base_steering.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --- Anchor pairs (exact text from plan) ---

anchor_tech_hi = (
    "extensive hands-on production experience in retrieval systems, "
    "vector search, embedding pipelines, and ranking infrastructure "
    "directly matching the technical requirements of the role"
)

anchor_tech_lo = (
    "background primarily in adjacent areas with limited direct "
    "exposure to retrieval, ranking, or vector database systems "
    "in production environments"
)

anchor_career_hi = (
    "clear trajectory of senior engineering ownership, end-to-end "
    "system design responsibility, and demonstrated measurable impact "
    "across multiple years at product companies"
)

anchor_career_lo = (
    "early career profile or primarily supporting role without "
    "demonstrated independent system ownership or technical decision "
    "making authority"
)

anchor_behav_hi = (
    "actively engaged on the platform with consistent recent activity, "
    "responsive to recruiter outreach, clearly available and currently "
    "in the job market"
)

anchor_behav_lo = (
    "minimal platform activity with low responsiveness rate, "
    "availability uncertain, likely not actively pursuing new "
    "opportunities right now"
)

# --- Synthetic candidate resume section (worked example from plan) ---

RESUME_SECTION_TEXT = (
    "Built two-stage FAISS retrieval pipeline at Meesho for 50M product "
    "catalog. P99 latency reduced from 340ms to 22ms. End-to-end ownership "
    "of ML infrastructure. Promoted to Senior MLE in 18 months."
)

# --- Default scores (test 1B) ---

S_TECH = 0.88
S_CAREER = 0.81
S_BEHAV = 0.19

# --- Hyperparameters ---

BETA = 0.55
MAX_LENGTH = 20
TEMPERATURE = 0.85
TOP_P = 0.92
DO_SAMPLE = True

DIMENSIONS = ("tech", "career", "behav")

ANCHOR_TEXTS: dict[str, tuple[str, str]] = {
    "tech": (anchor_tech_lo, anchor_tech_hi),
    "career": (anchor_career_lo, anchor_career_hi),
    "behav": (anchor_behav_lo, anchor_behav_hi),
}

# --- Experiment matrix (9 cases from plan) ---

EXPERIMENT_MATRIX: list[dict[str, Any]] = [
    {
        "id": "1A",
        "s_tech": 0.88,
        "s_career": 0.81,
        "s_behav": 0.65,
        "beta": 0.55,
        "expected_direction": "All three clauses positive",
    },
    {
        "id": "1B",
        "s_tech": 0.88,
        "s_career": 0.81,
        "s_behav": 0.19,
        "beta": 0.55,
        "expected_direction": "Two positive, one concern",
    },
    {
        "id": "1C",
        "s_tech": 0.30,
        "s_career": 0.40,
        "s_behav": 0.65,
        "beta": 0.55,
        "expected_direction": "Two concerns, one positive",
    },
    {
        "id": "2A",
        "s_tech": 0.88,
        "s_career": 0.81,
        "s_behav": 0.65,
        "beta": 0.40,
        "expected_direction": "More anchor language, less candidate detail",
    },
    {
        "id": "2B",
        "s_tech": 0.88,
        "s_career": 0.81,
        "s_behav": 0.65,
        "beta": 0.70,
        "expected_direction": "More candidate detail, weaker steering",
    },
    {
        "id": "3A",
        "s_tech": 0.50,
        "s_career": 0.50,
        "s_behav": 0.50,
        "beta": 0.55,
        "expected_direction": "All clauses neutral midpoint",
    },
]


def load_resume_from_json(path: Path | str, candidate_id: str) -> str:
    """Extract resume text from a candidate JSON file (read-only, no repo imports)."""
    json_path = Path(path)
    with json_path.open(encoding="utf-8") as f:
        candidates = json.load(f)

    for candidate in candidates:
        if candidate.get("candidate_id") != candidate_id:
            continue

        parts: list[str] = []
        profile = candidate.get("profile") or {}
        if headline := profile.get("headline"):
            parts.append(str(headline))
        if summary := profile.get("summary"):
            parts.append(str(summary))

        for role in candidate.get("career_history") or []:
            if description := role.get("description"):
                parts.append(str(description))

        if not parts:
            raise ValueError(f"No resume text found for candidate {candidate_id}")
        return " ".join(parts)

    raise ValueError(f"Candidate {candidate_id} not found in {json_path}")
