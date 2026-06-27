"""Pool routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.deps import executor_dep, job_store_dep, job_to_response, settings_dep
from backend.models.common import JobType
from backend.models.pools import (
    CreatePoolRequest,
    IndexJobResponse,
    PoolListResponse,
    PoolResponse,
    UploadCandidatesResponse,
)
from backend.services.pool_service import (
    create_pool,
    get_pool_or_404,
    list_pools,
    pool_to_response,
    save_candidates_upload,
)
from backend.services.readiness import check_readiness
from backend.settings import Settings
from backend.workers.executor import JobExecutor

router = APIRouter(prefix="/pools", tags=["pools"])


@router.post("", response_model=PoolResponse, status_code=201)
def create_pool_route(
    body: CreatePoolRequest,
    settings: Settings = Depends(settings_dep),
) -> PoolResponse:
    paths = create_pool(settings, body.name, body.description)
    return PoolResponse(**pool_to_response(paths))


@router.get("", response_model=PoolListResponse)
def list_pools_route(settings: Settings = Depends(settings_dep)) -> PoolListResponse:
    pools = [PoolResponse(**p) for p in list_pools(settings)]
    return PoolListResponse(pools=pools)


@router.get("/{pool_id}", response_model=PoolResponse)
def get_pool_route(
    pool_id: str,
    settings: Settings = Depends(settings_dep),
) -> PoolResponse:
    paths = get_pool_or_404(settings, pool_id)
    return PoolResponse(**pool_to_response(paths))


@router.post("/{pool_id}/candidates", response_model=UploadCandidatesResponse)
async def upload_candidates_route(
    pool_id: str,
    file: UploadFile = File(...),
    settings: Settings = Depends(settings_dep),
) -> UploadCandidatesResponse:
    paths = get_pool_or_404(settings, pool_id)
    count = await save_candidates_upload(paths, file, settings.candidate_id_pattern)
    return UploadCandidatesResponse(pool_id=pool_id, candidate_count=count)


@router.post("/{pool_id}/index", response_model=IndexJobResponse, status_code=202)
def index_pool_route(
    pool_id: str,
    settings: Settings = Depends(settings_dep),
    job_store=Depends(job_store_dep),
    executor: JobExecutor = Depends(executor_dep),
) -> IndexJobResponse:
    paths = get_pool_or_404(settings, pool_id)
    if not paths.candidates_jsonl.exists():
        raise HTTPException(status_code=400, detail="Upload candidates before indexing")

    readiness = check_readiness()
    if not readiness.checks.get("instructor_onnx"):
        raise HTTPException(status_code=503, detail="INSTRUCTOR ONNX artifacts not ready")
    if not readiness.checks.get("cuda_available"):
        raise HTTPException(status_code=503, detail="CUDA not available for indexing")

    record = job_store.create_job(job_type=JobType.index, pool_id=pool_id)
    executor.submit_index_job(record.job_id, pool_id)
    return IndexJobResponse(job_id=record.job_id, pool_id=pool_id, status="queued")
