#!/usr/bin/env python3
"""Online per-candidate resume-base steering experiment → results/output.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import (
    BETA,
    DIMENSIONS,
    RESUME_SECTION_TEXT,
    S_BEHAV,
    S_CAREER,
    S_TECH,
    load_resume_from_json,
)
from decode import gpt2_decode
from encode import LANGVAE_HF_ID, LATENT_DIM, langvae_encode
from steer import compute_final_vectors
from validate_decode import validate_roundtrip

VECTORS_DIR = _ROOT / "vectors"
RESULTS_DIR = _ROOT / "results"


def load_anchors(vectors_dir: Path = VECTORS_DIR) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    anchors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dim in DIMENSIONS:
        v_lo = np.load(vectors_dir / f"v_anch_{dim}_lo.npy")
        v_hi = np.load(vectors_dir / f"v_anch_{dim}_hi.npy")
        anchors[dim] = (v_lo, v_hi)
    return anchors


def build_reasoning(clauses: dict[str, str]) -> str:
    parts = [clauses[dim].strip().rstrip(".") for dim in DIMENSIONS]
    return ". ".join(parts) + "."


def run_pipeline(
    resume_text: str,
    s_tech: float,
    s_career: float,
    s_behav: float,
    beta: float,
    vectors_dir: Path = VECTORS_DIR,
) -> dict[str, Any]:
    anchors = load_anchors(vectors_dir)
    v_candidate = langvae_encode(resume_text)
    scores = {"tech": s_tech, "career": s_career, "behav": s_behav}
    v_final = compute_final_vectors(v_candidate, anchors, scores, beta)

    clauses = {dim: gpt2_decode(v_final[dim]) for dim in DIMENSIONS}
    reasoning = build_reasoning(clauses)

    return {
        "scores": {"s_tech": s_tech, "s_career": s_career, "s_behav": s_behav},
        "beta": beta,
        "resume_section_text": resume_text,
        "clauses": clauses,
        "reasoning": reasoning,
        "vector_dim": int(v_final["tech"].shape[0]),
        "langvae_checkpoint": LANGVAE_HF_ID,
    }


def resolve_resume_text(args: argparse.Namespace) -> str:
    if args.resume:
        return args.resume.strip()
    if args.resume_json and args.candidate_id:
        return load_resume_from_json(args.resume_json, args.candidate_id)
    return RESUME_SECTION_TEXT


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 1 resume-base steering experiment")
    parser.add_argument("--s-tech", type=float, default=S_TECH)
    parser.add_argument("--s-career", type=float, default=S_CAREER)
    parser.add_argument("--s-behav", type=float, default=S_BEHAV)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--resume", type=str, default=None, help="Override resume section text")
    parser.add_argument("--resume-json", type=Path, default=None, help="Path to candidates JSON")
    parser.add_argument(
        "--candidate-id",
        type=str,
        default=None,
        help="Candidate ID to load from --resume-json",
    )
    args = parser.parse_args()

    passed, total = validate_roundtrip()
    if passed < total:
        print(
            f"WARNING: LangVAE gate failed ({passed}/{total}). "
            "Output may be unrelated. Run validate_decode.py on Python 3.11 "
            "with requirements-pinned.txt.",
            file=sys.stderr,
        )

    resume_text = resolve_resume_text(args)
    output = run_pipeline(
        resume_text=resume_text,
        s_tech=args.s_tech,
        s_career=args.s_career,
        s_behav=args.s_behav,
        beta=args.beta,
    )
    output["decode_gate_passed"] = passed == total

    for dim in DIMENSIONS:
        print(f"=== CLAUSE {dim.upper()} ===")
        print(output["clauses"][dim])
        print()
    print("=== FINAL REASONING ===")
    print(output["reasoning"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "output.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
