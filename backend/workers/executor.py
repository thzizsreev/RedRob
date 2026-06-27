"""Background job execution."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from backend.services.job_store import InMemoryJobStore, JobStatus, get_job_store
from backend.services.pool_paths import resolve_pool
from backend.settings import Settings, get_settings


class JobExecutor:
    def __init__(self, settings: Settings, job_store: InMemoryJobStore) -> None:
        self._settings = settings
        self._job_store = job_store
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def _run_with_gpu_lock(self, job_id: str, fn: Callable[[], dict]) -> None:
        store = self._job_store
        if not store.acquire_gpu_lock():
            store.update(
                job_id,
                status=JobStatus.failed,
                error="Could not acquire GPU job lock",
                finished=True,
            )
            return
        try:
            result = fn()
            store.update(
                job_id,
                status=JobStatus.completed,
                result=result,
                progress="completed",
                finished=True,
            )
        except Exception as exc:
            store.update(
                job_id,
                status=JobStatus.failed,
                error=str(exc),
                finished=True,
            )
        finally:
            store.release_gpu_lock()

    def _submit(self, job_id: str, fn: Callable[[], dict]) -> Future:
        if self._settings.sync_jobs:
            fut: Future = Future()
            try:
                self._run_with_gpu_lock(job_id, fn)
                fut.set_result(None)
            except Exception as exc:
                self._job_store.update(
                    job_id,
                    status=JobStatus.failed,
                    error=str(exc),
                    finished=True,
                )
                fut.set_exception(exc)
            return fut
        return self._executor.submit(self._run_with_gpu_lock, job_id, fn)

    def submit_index_job(self, job_id: str, pool_id: str, random_seed: int | None = None) -> Future:
        paths = resolve_pool(pool_id, self._settings.api_pools_root)

        def task() -> dict:
            from tracks.instructor.core.config import STAGE1_RANDOM_SEED

            from backend.services.index_service import run_index

            seed = random_seed if random_seed is not None else STAGE1_RANDOM_SEED
            return run_index(
                job_id=job_id,
                paths=paths,
                settings=self._settings,
                job_store=self._job_store,
                random_seed=seed,
            )

        return self._submit(job_id, task)

    def submit_ranking_job(
        self,
        job_id: str,
        pool_id: str,
        config_path: Path | None = None,
        random_seed: int | None = None,
    ) -> Future:
        paths = resolve_pool(pool_id, self._settings.api_pools_root)

        def task() -> dict:
            from backend.services.ranking_service import run_ranking

            return run_ranking(
                job_id=job_id,
                paths=paths,
                settings=self._settings,
                job_store=self._job_store,
                config_path=config_path,
                random_seed=random_seed,
            )

        return self._submit(job_id, task)


_executor: JobExecutor | None = None


def get_executor() -> JobExecutor:
    global _executor
    if _executor is None:
        _executor = JobExecutor(get_settings(), get_job_store())
    return _executor


def init_executor(settings: Settings, job_store: InMemoryJobStore) -> JobExecutor:
    global _executor
    _executor = JobExecutor(settings, job_store)
    return _executor


def shutdown_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
