"""Stage 5 composite scoring orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import polars as pl

from tracks.instructor.stage5.config import Stage5Config, load_stage5_config
from tracks.instructor.stage5.io import (
    join_scoring_inputs,
    load_stage4_reranked,
    warn_input_count,
    write_stage5_outputs,
)
from tracks.instructor.stage5.layers import apply_scoring
from tracks.instructor.stage5.reasoning import compose_reasoning
from tracks.instructor.stage5.validate import validate_submission_csv


@dataclass(frozen=True)
class Stage5Result:
    input_count: int
    output_count: int
    score_min: float
    score_max: float
    score_mean: float
    elapsed_seconds: float
    csv_path: Path
    output_dir: Path


def run(
    *,
    stage4_path: Path,
    output_dir: Path,
    config_path: Path,
) -> Stage5Result:
    start = perf_counter()
    config = load_stage5_config(config_path)

    stage4_df = load_stage4_reranked(stage4_path)
    warn_input_count(stage4_df.height)

    joined = join_scoring_inputs(stage4_df, config)
    scored = apply_scoring(joined, config)

    ranked = scored.sort(["final_score", "candidate_id"], descending=[True, True])
    top_n = min(config.top_n, ranked.height)
    top = ranked.head(top_n).with_columns(
        pl.int_range(1, top_n + 1).alias("rank"),
    )

    submission_rows: list[dict] = []
    for row in top.iter_rows(named=True):
        submission_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],
                "score": round(float(row["final_score"]), 6),
                "reasoning": compose_reasoning(row),
            }
        )

    scores = top["final_score"]
    summary = {
        "input_count": stage4_df.height,
        "output_count": top_n,
        "team_id": config.team_id,
        "score_min": float(scores.min()) if top_n else 0.0,
        "score_max": float(scores.max()) if top_n else 0.0,
        "score_mean": float(scores.mean()) if top_n else 0.0,
        "elapsed_seconds": round(perf_counter() - start, 3),
    }

    csv_path = write_stage5_outputs(
        output_dir,
        config.team_id,
        scored,
        top,
        submission_rows,
        summary,
    )
    validate_submission_csv(csv_path, expected_rows=top_n)

    elapsed = perf_counter() - start
    return Stage5Result(
        input_count=stage4_df.height,
        output_count=top_n,
        score_min=summary["score_min"],
        score_max=summary["score_max"],
        score_mean=summary["score_mean"],
        elapsed_seconds=elapsed,
        csv_path=csv_path,
        output_dir=output_dir,
    )


def print_stage5_summary(result: Stage5Result) -> None:
    print("\n--- Stage 5 summary ---")
    print(f"Input:     {result.input_count:,}")
    print(f"Output:    {result.output_count:,}")
    print(f"Score min: {result.score_min:.4f}")
    print(f"Score max: {result.score_max:.4f}")
    print(f"Score avg: {result.score_mean:.4f}")
    print(f"Elapsed:   {result.elapsed_seconds:.2f}s")
    print(f"\nWrote submission to {result.csv_path}")
