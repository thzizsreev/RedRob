"""Hardcoded strings, scores, and hyperparameters from vector_steering_plan_3_sonar.md."""

from __future__ import annotations

# --- Template library ---

template_tech = {
    "high": (
        "Strong production-grade technical alignment with the retrieval and "
        "ranking requirements of the role, backed by hands-on engineering "
        "ownership at meaningful scale."
    ),
    "mid": (
        "Partial technical overlap with the role's retrieval and ranking "
        "focus, with some relevant exposure but gaps in depth or breadth "
        "of production experience."
    ),
    "low": (
        "Limited direct technical alignment with the role's core retrieval "
        "and embedding systems requirements based on the available profile."
    ),
}

template_career = {
    "high": (
        "Senior engineering trajectory with demonstrated end-to-end "
        "ownership, measurable impact, and a clear product-company "
        "background matching the seniority target."
    ),
    "mid": (
        "Developing seniority with some ownership signals but not yet "
        "at the full independent decision-making depth the role requires."
    ),
    "low": (
        "Early career profile or primarily supporting role history "
        "without the seniority and ownership depth the position demands."
    ),
}

template_behav = {
    "high": (
        "Active on platform with strong engagement signals and low "
        "friction expected for recruiter outreach and response."
    ),
    "mid": (
        "Moderate platform engagement with some responsiveness signals, "
        "though availability may need confirmation."
    ),
    "low": (
        "Minimal recent platform activity and low responsiveness signals "
        "suggest availability is uncertain and outreach may face friction."
    ),
}

# --- Anchor pairs ---

anchor_tech_hi = (
    "hands-on production engineer who has built and owned retrieval systems, "
    "vector databases, and embedding pipelines at real scale with measurable "
    "latency and quality outcomes"
)

anchor_tech_lo = (
    "candidate with surface-level or theoretical exposure to retrieval concepts "
    "without direct ownership of production retrieval or ranking systems"
)

anchor_career_hi = (
    "senior individual contributor with full ownership of complex ML systems, "
    "clear promotion trajectory, and cross-functional impact at product companies"
)

anchor_career_lo = (
    "junior or mid-level engineer in a supporting capacity with limited "
    "independent scope and no clear ownership of significant systems"
)

anchor_behav_hi = (
    "candidate who is actively engaged, logged in recently, responds quickly "
    "to recruiter messages, and has low notice period"
)

anchor_behav_lo = (
    "candidate who has not logged in for an extended period, rarely responds "
    "to recruiter outreach, and shows no recent job-seeking activity"
)

# --- Candidate resume sections (hardcoded test case) ---

resume_tech_text = (
    "Built two-stage FAISS retrieval pipeline at Meesho for 50M product catalog. "
    "P99 latency reduced from 340ms to 22ms. End-to-end ownership of the system."
)

resume_career_text = (
    "Promoted to Senior MLE in 18 months. Led a team of 4 engineers. "
    "Full ownership of ML infrastructure decisions."
)

resume_behav_text = (
    "47 days since last login. Response rate 0.18. Notice period 30 days."
)

# --- Scores (hardcoded test case) ---

s_tech = 0.88
s_career = 0.81
s_behav = 0.19

# --- Fixed hyperparameters ---

GAMMA = 0.55
DELTA = 0.30
MAX_SEQ_LEN = 64
BEAM_SIZE = 5
TARGET_LANG = "eng_Latn"
SOURCE_LANG = "eng_Latn"

VECTOR_DIM = 1024
SONAR_ENCODER = "text_sonar_basic_encoder"
SONAR_DECODER = "text_sonar_basic_decoder"


def bucket(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "mid"
    return "low"
