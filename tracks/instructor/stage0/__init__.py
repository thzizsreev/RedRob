"""Stage 0 — offline vector precompute (FAISS index + JD query vector)."""

from tracks.instructor.stage0.precompute import load_candidates_json, run_precompute

__all__ = ["load_candidates_json", "run_precompute"]
