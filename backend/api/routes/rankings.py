"""Ranking and job routes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from backend.api.deps import executor_dep, job_store_dep, job_to_response, settings_dep
from backend.models.common import JobResponse, JobStatus, JobType
from backend.models.rankings import (
    CreateRankingRequest,
    RankingJobResponse,
    RankingResultItem,
    RankingResultsResponse,
)
from backend.services.pool_paths import resolve_pool
from backend.services.pool_service import get_pool_or_404
from backend.services.readiness import check_readiness
from backend.settings import Settings
from backend.workers.executor import JobExecutor

router = APIRouter(tags=["rankings"])


@router.post("/rankings", response_model=RankingJobResponse, status_code=202)
def create_ranking_route(
    body: CreateRankingRequest,
    settings: Settings = Depends(settings_dep),
    job_store=Depends(job_store_dep),
    executor: JobExecutor = Depends(executor_dep),
) -> RankingJobResponse:
    get_pool_or_404(settings, body.pool_id)

    readiness = check_readiness()
    if not readiness.ready:
        raise HTTPException(
            status_code=503,
            detail={"message": "System not ready", "checks": readiness.checks},
        )

    config_path = None
    if body.config_path:
        config_path = Path(body.config_path)
        if not config_path.is_absolute():
            config_path = (settings.default_config_path.parent / body.config_path).resolve()
        if not config_path.exists():
            raise HTTPException(status_code=400, detail=f"Config not found: {config_path}")

    record = job_store.create_job(
        job_type=JobType.ranking,
        pool_id=body.pool_id,
        config_path=str(config_path) if config_path else None,
        random_seed=body.random_seed,
    )
    executor.submit_ranking_job(
        record.job_id,
        body.pool_id,
        config_path=config_path,
        random_seed=body.random_seed,
    )
    return RankingJobResponse(job_id=record.job_id, pool_id=body.pool_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_route(job_id: str, job_store=Depends(job_store_dep)) -> JobResponse:
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobResponse(**job_to_response(record))


def _load_ranking_results(job_id: str, job_store, settings: Settings) -> RankingResultsResponse:
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if record.type != JobType.ranking:
        raise HTTPException(status_code=400, detail="Job is not a ranking job")
    if record.status != JobStatus.completed:
        raise HTTPException(
            status_code=409,
            detail=f"Ranking job not completed (status={record.status.value})",
        )

    csv_path = None
    if record.result:
        csv_path = record.result.get("final_csv_path") or record.result.get("stage5_csv")

    if not csv_path or not Path(csv_path).exists():
        paths = resolve_pool(record.pool_id, settings.api_pools_root)
        stage5_dir = paths.stage5
        if stage5_dir.exists():
            csv_files = list(stage5_dir.glob("*.csv"))
            if csv_files:
                csv_path = str(csv_files[0])

    if not csv_path or not Path(csv_path).exists():
        raise HTTPException(status_code=404, detail="Ranking results CSV not found")

    items: list[RankingResultItem] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                RankingResultItem(
                    candidate_id=row["candidate_id"],
                    rank=int(row["rank"]),
                    score=float(row["score"]),
                    reasoning=row["reasoning"],
                )
            )

    summary: dict = {}
    summary_path = Path(csv_path).parent / "stage5_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return RankingResultsResponse(
        job_id=job_id,
        pool_id=record.pool_id,
        items=items,
        summary=summary,
    )


@router.get("/rankings/{job_id}/results", response_model=RankingResultsResponse)
def get_ranking_results_route(
    job_id: str,
    settings: Settings = Depends(settings_dep),
    job_store=Depends(job_store_dep),
) -> RankingResultsResponse:
    return _load_ranking_results(job_id, job_store, settings)


@router.get("/rankings/{job_id}/results.csv")
def download_ranking_csv_route(
    job_id: str,
    settings: Settings = Depends(settings_dep),
    job_store=Depends(job_store_dep),
) -> FileResponse:
    _load_ranking_results(job_id, job_store, settings)
    record = job_store.get(job_id)
    assert record is not None
    csv_path = record.result.get("final_csv_path") if record.result else None
    if not csv_path or not Path(csv_path).exists():
        paths = resolve_pool(record.pool_id, settings.api_pools_root)
        csv_files = list(paths.stage5.glob("*.csv"))
        if not csv_files:
            raise HTTPException(status_code=404, detail="CSV not found")
        csv_path = str(csv_files[0])
    return FileResponse(csv_path, media_type="text/csv", filename=Path(csv_path).name)
