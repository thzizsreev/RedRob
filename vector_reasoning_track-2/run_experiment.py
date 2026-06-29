#!/usr/bin/env python3
"""SONAR encode → steer → decode experiment → results/output.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import (
    DELTA,
    GAMMA,
    SONAR_DECODER,
    SONAR_ENCODER,
    VECTOR_DIM,
    bucket,
    resume_behav_text,
    resume_career_text,
    resume_tech_text,
    s_behav,
    s_career,
    s_tech,
)
from decode import sonar_decode
from encode import sonar_encode

VECTORS_DIR = _ROOT / "vectors"
RESULTS_DIR = _ROOT / "results"


def main() -> None:
    bucket_tech = bucket(s_tech)
    bucket_career = bucket(s_career)
    bucket_behav = bucket(s_behav)

    v_tmpl_tech = np.load(VECTORS_DIR / f"v_tmpl_tech_{bucket_tech}.npy")
    v_tmpl_career = np.load(VECTORS_DIR / f"v_tmpl_career_{bucket_career}.npy")
    v_tmpl_behav = np.load(VECTORS_DIR / f"v_tmpl_behav_{bucket_behav}.npy")

    v_anch_tech_hi = np.load(VECTORS_DIR / "v_anch_tech_hi.npy")
    v_anch_tech_lo = np.load(VECTORS_DIR / "v_anch_tech_lo.npy")
    v_anch_career_hi = np.load(VECTORS_DIR / "v_anch_career_hi.npy")
    v_anch_career_lo = np.load(VECTORS_DIR / "v_anch_career_lo.npy")
    v_anch_behav_hi = np.load(VECTORS_DIR / "v_anch_behav_hi.npy")
    v_anch_behav_lo = np.load(VECTORS_DIR / "v_anch_behav_lo.npy")

    v_cand_tech = sonar_encode(resume_tech_text)
    v_cand_career = sonar_encode(resume_career_text)
    v_cand_behav = sonar_encode(resume_behav_text)

    v_base_tech = GAMMA * v_tmpl_tech + (1 - GAMMA) * v_cand_tech
    v_base_career = GAMMA * v_tmpl_career + (1 - GAMMA) * v_cand_career
    v_base_behav = GAMMA * v_tmpl_behav + (1 - GAMMA) * v_cand_behav

    v_steer_tech = (1 - s_tech) * v_anch_tech_lo + s_tech * v_anch_tech_hi
    v_steer_career = (1 - s_career) * v_anch_career_lo + s_career * v_anch_career_hi
    v_steer_behav = (1 - s_behav) * v_anch_behav_lo + s_behav * v_anch_behav_hi

    v_final_tech = (1 - DELTA) * v_base_tech + DELTA * v_steer_tech
    v_final_career = (1 - DELTA) * v_base_career + DELTA * v_steer_career
    v_final_behav = (1 - DELTA) * v_base_behav + DELTA * v_steer_behav

    clause_tech = sonar_decode(v_final_tech)
    clause_career = sonar_decode(v_final_career)
    clause_behav = sonar_decode(v_final_behav)

    reasoning = (
        clause_tech.strip().rstrip(".") + ". "
        + clause_career.strip().rstrip(".") + ". "
        + clause_behav.strip().rstrip(".") + "."
    )

    print("=== CLAUSE TECH ===")
    print(clause_tech)
    print()
    print("=== CLAUSE CAREER ===")
    print(clause_career)
    print()
    print("=== CLAUSE BEHAV ===")
    print(clause_behav)
    print()
    print("=== FINAL REASONING ===")
    print(reasoning)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "scores": {
            "s_tech": s_tech,
            "s_career": s_career,
            "s_behav": s_behav,
        },
        "buckets": {
            "bucket_tech": bucket_tech,
            "bucket_career": bucket_career,
            "bucket_behav": bucket_behav,
        },
        "clauses": {
            "clause_tech": clause_tech,
            "clause_career": clause_career,
            "clause_behav": clause_behav,
        },
        "reasoning": reasoning,
        "encoder": SONAR_ENCODER,
        "decoder": SONAR_DECODER,
        "vector_dim": VECTOR_DIM,
    }
    output_path = RESULTS_DIR / "output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Saved to results/output.json")


if __name__ == "__main__":
    main()
