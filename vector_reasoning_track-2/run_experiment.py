#!/usr/bin/env python3
"""Online per-candidate dimensional steering experiment → results/output.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compose import compose_clause
from constants import (
    DELTA,
    GAMMA,
    bucket,
    resume_behav_text,
    resume_career_text,
    resume_tech_text,
    s_behav,
    s_career,
    s_tech,
    template_behav,
    template_career,
    template_tech,
)
from decode import langvae_decode
from encode import LANGVAE_HF_ID, langvae_encode
from validate_langvae import validate_roundtrip

VECTORS_DIR = _ROOT / "vectors"
RESULTS_DIR = _ROOT / "results"


def _build_reasoning(
    prompt_tech: str,
    clause_tech: str,
    prompt_career: str,
    clause_career: str,
    prompt_behav: str,
    clause_behav: str,
) -> str:
    return (
        prompt_tech.rstrip(".") + " " + clause_tech.strip().rstrip(".") + ". "
        + prompt_career.rstrip(".") + " " + clause_career.strip().rstrip(".") + ". "
        + prompt_behav.rstrip(".") + " " + clause_behav.strip().rstrip(".") + "."
    )


def run_vector_pipeline() -> tuple[dict[str, np.ndarray], dict[str, str]]:
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

    v_cand_tech = langvae_encode(resume_tech_text)
    v_cand_career = langvae_encode(resume_career_text)
    v_cand_behav = langvae_encode(resume_behav_text)

    v_base_tech = GAMMA * v_tmpl_tech + (1 - GAMMA) * v_cand_tech
    v_base_career = GAMMA * v_tmpl_career + (1 - GAMMA) * v_cand_career
    v_base_behav = GAMMA * v_tmpl_behav + (1 - GAMMA) * v_cand_behav

    v_steer_tech = (1 - s_tech) * v_anch_tech_lo + s_tech * v_anch_tech_hi
    v_steer_career = (1 - s_career) * v_anch_career_lo + s_career * v_anch_career_hi
    v_steer_behav = (1 - s_behav) * v_anch_behav_lo + s_behav * v_anch_behav_hi

    v_final = {
        "tech": (1 - DELTA) * v_base_tech + DELTA * v_steer_tech,
        "career": (1 - DELTA) * v_base_career + DELTA * v_steer_career,
        "behav": (1 - DELTA) * v_base_behav + DELTA * v_steer_behav,
    }
    buckets = {
        "bucket_tech": bucket_tech,
        "bucket_career": bucket_career,
        "bucket_behav": bucket_behav,
    }
    return v_final, buckets


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector reasoning steering experiment")
    parser.add_argument(
        "--decode",
        choices=("template_hybrid", "langvae"),
        default="template_hybrid",
        help="Decode mode: template_hybrid (reliable) or langvae (requires validate_langvae.py gate)",
    )
    args = parser.parse_args()

    v_final, buckets = run_vector_pipeline()
    bucket_tech = buckets["bucket_tech"]
    bucket_career = buckets["bucket_career"]
    bucket_behav = buckets["bucket_behav"]

    prompt_tech = template_tech[bucket_tech]
    prompt_career = template_career[bucket_career]
    prompt_behav = template_behav[bucket_behav]

    if args.decode == "template_hybrid":
        clause_tech = compose_clause("tech", s_tech)
        clause_career = compose_clause("career", s_career)
        clause_behav = compose_clause("behav", s_behav)
        decoder_meta = {"decoder": "template_hybrid"}
    else:
        passed, total = validate_roundtrip()
        if passed < total:
            print(
                f"WARNING: LangVAE gate failed ({passed}/{total}). "
                "Output may be unrelated. Run validate_langvae.py on Python 3.11 "
                "with requirements-pinned.txt, or use --decode template_hybrid.",
                file=sys.stderr,
            )
        clause_tech = langvae_decode(v_final["tech"])
        clause_career = langvae_decode(v_final["career"])
        clause_behav = langvae_decode(v_final["behav"])
        decoder_meta = {
            "decoder": "langvae",
            "langvae_checkpoint": LANGVAE_HF_ID,
            "langvae_gate_passed": passed == total,
        }

    reasoning = _build_reasoning(
        prompt_tech,
        clause_tech,
        prompt_career,
        clause_career,
        prompt_behav,
        clause_behav,
    )

    print("=== PROMPT TECH ===")
    print(prompt_tech)
    print("=== CLAUSE TECH ===")
    print(clause_tech)
    print()
    print("=== PROMPT CAREER ===")
    print(prompt_career)
    print("=== CLAUSE CAREER ===")
    print(clause_career)
    print()
    print("=== PROMPT BEHAV ===")
    print(prompt_behav)
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
        "buckets": buckets,
        "prompts": {
            "prompt_tech": prompt_tech,
            "prompt_career": prompt_career,
            "prompt_behav": prompt_behav,
        },
        "clauses": {
            "clause_tech": clause_tech,
            "clause_career": clause_career,
            "clause_behav": clause_behav,
        },
        "reasoning": reasoning,
        "vector_dim": int(v_final["tech"].shape[0]),
        **decoder_meta,
    }
    output_path = RESULTS_DIR / "output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Saved to results/output.json")


if __name__ == "__main__":
    main()
