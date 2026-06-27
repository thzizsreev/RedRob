"""FastAPI dependencies."""

from __future__ import annotations

from backend.services.job_store import InMemoryJobStore, JobRecord, get_job_store
from backend.settings import Settings, get_settings
from backend.workers.executor import JobExecutor, get_executor


def settings_dep() -> Settings:
    return get_settings()


def job_store_dep() -> InMemoryJobStore:
    return get_job_store()


def executor_dep() -> JobExecutor:
    return get_executor()


def job_to_response(record: JobRecord) -> dict:
    timings = None
    if record.result and "timings" in record.result:
        timings = record.result["timings"]
    return {
        "job_id": record.job_id,
        "type": record.type,
        "status": record.status,
        "pool_id": record.pool_id,
        "progress": record.progress,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
        "result": record.result,
        "timings": timings,
    }
