"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.api.routes import health, pools, rankings
from backend.models.common import ErrorResponse
from backend.services.job_store import get_job_store
from backend.settings import get_settings
from backend.workers.executor import init_executor, shutdown_executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    job_store = get_job_store()
    init_executor(settings, job_store)
    yield
    shutdown_executor()


app = FastAPI(
    title="RedRob Ranking API",
    description="REST API for RedRob candidate ranking pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

api_prefix = "/api/v1"
app.include_router(health.router, prefix=api_prefix)
app.include_router(pools.router, prefix=api_prefix)
app.include_router(rankings.router, prefix=api_prefix)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=str(exc.detail)).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail=str(exc), code="internal_error").model_dump(),
    )
