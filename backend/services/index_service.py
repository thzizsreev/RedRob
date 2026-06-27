"""Stage 0 + cluster indexing orchestration."""

from __future__ import annotations

from backend.services.job_store import InMemoryJobStore, JobStatus
from backend.services.pool_paths import PoolPaths
from backend.services.pool_service import mark_pool_indexed
from backend.settings import Settings


def run_index(
    *,
    job_id: str,
    paths: PoolPaths,
    settings: Settings,
    job_store: InMemoryJobStore,
    random_seed: int,
) -> dict:
    from tracks.instructor.core.onnx_embedder import load_embedder, unload_embedder
    from tracks.instructor.stage0.cluster_precompute import run_cluster_precompute
    from tracks.instructor.stage0.precompute import run_precompute
    from tracks.instructor.stage0.skill_precompute import run_skill_precompute
    from tracks.instructor.stage0.stage3_query_precompute import run_stage3_query_precompute

    if not paths.candidates_jsonl.exists():
        raise FileNotFoundError("Candidates file missing; upload candidates first")

    config_path = settings.default_config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    job_store.update(job_id, status=JobStatus.running, progress="encoding", started=True)

    model = load_embedder()
    try:
        records = run_precompute(paths.candidates_jsonl, model, paths.stage0)
        job_store.update(job_id, progress="skill_scores")
        run_skill_precompute(records, paths.stage0, config_path)
        job_store.update(job_id, progress="query_vectors")
        run_stage3_query_precompute(model, paths.stage0, config_path)
    finally:
        unload_embedder(model)

    job_store.update(job_id, progress="clustering")
    run_cluster_precompute(
        paths.stage0,
        paths.stage1,
        random_seed=random_seed,
        overwrite=True,
    )

    mark_pool_indexed(paths)
    job_store.update(job_id, progress="completed")

    return {
        "candidate_count": len(records),
        "stage0_path": str(paths.stage0),
        "stage1_path": str(paths.stage1),
    }
