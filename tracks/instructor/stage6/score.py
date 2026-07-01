"""Stage 6 orchestrator — 3-sentence reasoning for top-100."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

from tracks.instructor.stage6.config import Stage6Config, load_stage6_config
from tracks.instructor.stage6.io import (
    build_candidate_dicts,
    load_reasoning_lookup,
    load_reasoning_raw_cache,
    reasoning_rows_from_lookup,
    write_reasoning_lookup_from_submission,
    write_stage6_outputs,
)
from tracks.instructor.stage6.paraphrase_onnx import init_paraphraser, make_paraphrase_fn
from tracks.instructor.stage6.pool import process_candidates_parallel
from tracks.instructor.stage6.validate import validate_stage6_csv
from tracks.instructor.stage6.workers import resolve_worker_count


def _enforce_monotonic_scores(scores: list[float]) -> tuple[list[float], int]:
    if not scores:
        return scores, 0
    out = [scores[0]]
    corrections = 0
    for i in range(1, len(scores)):
        prev = out[i - 1]
        cur = scores[i]
        if cur > prev:
            corrections += 1
            out.append(prev)
        else:
            out.append(cur)
    return out, corrections


@dataclass(frozen=True)
class Stage6Result:
    input_count: int
    output_count: int
    worker_count: int
    elapsed_seconds: float
    csv_path: Path
    output_dir: Path
    summary: dict = field(default_factory=dict)


def run(
    *,
    config_path: Path,
    output_dir: Path | None = None,
) -> Stage6Result:
    start = perf_counter()
    config = load_stage6_config(config_path)
    if output_dir is not None:
        config = replace(config, output_dir=output_dir.resolve())

    top_df, candidates = build_candidate_dicts(config)
    raw_cache = load_reasoning_raw_cache(config.reasoning_raw_path)
    candidate_ids = [str(c["candidate_id"]) for c in candidates]

    reasoning_lookup = load_reasoning_lookup(config.reasoning_lookup_path)
    cached_rows = None
    if config.use_reasoning_lookup in ("auto", "true"):
        cached_rows = reasoning_rows_from_lookup(candidate_ids, reasoning_lookup)
        if config.use_reasoning_lookup == "true" and cached_rows is None:
            missing = [cid for cid in candidate_ids if cid not in reasoning_lookup]
            raise ValueError(
                f"use_reasoning_lookup=true but {len(missing)} candidate(s) missing "
                f"from {config.reasoning_lookup_path}. Examples: {sorted(missing)[:5]}"
            )

    worker_count, worker_rationale = resolve_worker_count(
        len(candidates),
        ort_intra_op_threads=config.ort_intra_op_threads,
        estimated_session_mb=config.estimated_session_mb,
        memory_reserve_ratio=config.memory_reserve_ratio,
        max_workers=config.max_workers,
    )

    paraphrase_start = perf_counter()
    if cached_rows is not None:
        print(
            f"Using reasoning lookup ({len(cached_rows)} rows) — skipping paraphrase."
        )
        reasoning_rows = cached_rows
        worker_count = 0
        worker_rationale = {**worker_rationale, "chosen": 0, "reasoning_lookup": True}
    else:
        print(f"Stage 6 worker count: {worker_count} ({worker_rationale})")
        init_paraphraser(config)
        paraphrase_fn = make_paraphrase_fn()
        reasoning_rows = process_candidates_parallel(
            candidates,
            raw_cache,
            paraphrase_fn,
            worker_count,
        )
    paraphrase_elapsed = perf_counter() - paraphrase_start

    reasoning_by_id = {str(r["candidate_id"]): r for r in reasoning_rows}

    sorted_rows = list(top_df.sort("rank").iter_rows(named=True))
    raw_scores = [float(row["final_score"]) for row in sorted_rows]
    clamped_scores, monotonic_corrections = _enforce_monotonic_scores(raw_scores)

    submission_rows: list[dict] = []
    for row, score in zip(sorted_rows, clamped_scores):
        cid = str(row["candidate_id"])
        reasoning = reasoning_by_id[cid]
        submission_rows.append(
            {
                "candidate_id": cid,
                "rank": int(row["rank"]),
                "score": round(score, 6),
                "reasoning": reasoning["reasoning"],
            }
        )

    elapsed = perf_counter() - start
    summary = {
        "input_count": len(candidates),
        "output_count": len(submission_rows),
        "worker_count": worker_count,
        "worker_rationale": worker_rationale,
        "paraphrase_seconds": round(paraphrase_elapsed, 3),
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(len(candidates) / paraphrase_elapsed, 3)
        if paraphrase_elapsed > 0
        else 0.0,
        "team_id": config.team_id,
        "monotonic_corrections": monotonic_corrections,
        "reasoning_lookup_used": cached_rows is not None,
    }

    csv_path = write_stage6_outputs(
        config.output_dir,
        config.team_id,
        submission_rows,
        reasoning_rows,
        summary,
    )

    if cached_rows is None:
        write_reasoning_lookup_from_submission(
            submission_rows,
            csv_path,
            config.reasoning_lookup_path,
        )

    validate_stage6_csv(
        csv_path,
        expected_rows=len(submission_rows),
        input_candidate_ids=set(reasoning_by_id.keys()),
    )

    return Stage6Result(
        input_count=len(candidates),
        output_count=len(submission_rows),
        worker_count=worker_count,
        elapsed_seconds=elapsed,
        csv_path=csv_path,
        output_dir=config.output_dir,
        summary=summary,
    )


def print_stage6_summary(result: Stage6Result) -> None:
    s = result.summary
    print(f"\nStage 6 complete: {result.output_count} candidates")
    print(f"  workers:    {result.worker_count}")
    print(f"  paraphrase: {s.get('paraphrase_seconds', 0):.2f}s")
    print(f"  total:      {result.elapsed_seconds:.2f}s")
    print(f"  throughput: {s.get('rows_per_second', 0):.3f} candidates/s")
    print(f"  CSV:        {result.csv_path}")
