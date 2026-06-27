"""Stages 1–5 ranking orchestration."""

from __future__ import annotations

from pathlib import Path

from backend.services.job_store import InMemoryJobStore, JobStatus
from backend.services.pool_paths import PoolPaths, check_index_readiness
from backend.settings import Settings


def run_ranking(
    *,
    job_id: str,
    paths: PoolPaths,
    settings: Settings,
    job_store: InMemoryJobStore,
    config_path: Path | None = None,
    random_seed: int | None = None,
) -> dict:
    from tracks.instructor.core.config import STAGE1_RANDOM_SEED
    from tracks.instructor.pipeline import RankingPipelineConfig, run_ranking_pipeline

    readiness = check_index_readiness(paths)
    if not readiness.get("indexed"):
        raise RuntimeError(
            "Pool is not indexed. Run POST /pools/{pool_id}/index first. "
            f"Checks: {readiness}"
        )

    if not paths.candidates_jsonl.exists():
        raise FileNotFoundError("Candidates file missing")

    resolved_config = config_path or settings.default_config_path
    if not resolved_config.exists():
        raise FileNotFoundError(f"Config not found: {resolved_config}")

    job_store.update(job_id, status=JobStatus.running, progress="ranking", started=True)

    cfg = RankingPipelineConfig(
        stage0_path=paths.stage0,
        stage1_path=paths.stage1,
        stage2_output_dir=paths.stage2,
        stage3_output_dir=paths.stage3,
        stage4_output_dir=paths.stage4,
        stage5_output_dir=paths.stage5,
        candidates_path=paths.candidates_jsonl,
        config_path=resolved_config,
        random_seed=random_seed if random_seed is not None else STAGE1_RANDOM_SEED,
        print_summaries=False,
    )

    result = run_ranking_pipeline(cfg, print_timing=False)

    timings = [
        {
            "stage": t.stage,
            "label": t.label,
            "elapsed_seconds": t.elapsed_seconds,
        }
        for t in result.timings
    ]

    return {
        "final_csv_path": str(result.final_csv_path),
        "total_elapsed_seconds": result.total_elapsed_seconds,
        "timings": timings,
        "stage1_filtered": len(result.stage1.result.filtered_ids),
        "stage2_survivors": result.stage2.survivor_count,
        "stage3_output": result.stage3.output_count,
        "stage4_output": result.stage4.output_count,
        "stage5_csv": str(result.final_csv_path),
    }
