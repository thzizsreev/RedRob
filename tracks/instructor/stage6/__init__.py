"""Stage 6 — 3-sentence reasoning builder with CPU ONNX paraphraser."""

from tracks.instructor.stage6.score import (
    Stage6Result,
    print_stage6_summary,
    run,
    run_from_ranking_csv,
)

__all__ = ["Stage6Result", "print_stage6_summary", "run", "run_from_ranking_csv"]
