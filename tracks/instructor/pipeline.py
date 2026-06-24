"""Stages 1–5 ranking pipeline — orchestrates filter → gate → retrieve → rerank → score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from tracks.instructor.core.config import STAGE1_RANDOM_SEED
from tracks.instructor.stage1 import run_stage1_filter
from tracks.instructor.stage1.pipeline import Stage1RunResult
from tracks.instructor.stage2 import print_stage2_summary, run as run_stage2
from tracks.instructor.stage2.gate import Stage2Result
from tracks.instructor.stage3 import print_stage3_summary, run as run_stage3
from tracks.instructor.stage3.retrieve import Stage3Result
from tracks.instructor.stage4 import print_stage4_summary, run as run_stage4
from tracks.instructor.stage4.rerank import Stage4Result
from tracks.instructor.stage5 import print_stage5_summary, run as run_stage5
from tracks.instructor.stage5.score import Stage5Result
from tracks.shared.paths import (
    CANDIDATES_JSONL_PATH,
    ROOT_DIR,
    RUNTIME_STAGE0_DIR,
    RUNTIME_STAGE1_DIR,
    RUNTIME_STAGE2_DIR,
    RUNTIME_STAGE3_DIR,
    RUNTIME_STAGE4_DIR,
    RUNTIME_STAGE5_DIR,
)

__all__ = [
    "RankingPipelineConfig",
    "RankingPipelineResult",
    "StageTiming",
    "run_ranking_pipeline",
    "print_pipeline_timing",
]

_STAGE_LABELS: dict[int, str] = {
    1: "filter",
    2: "gate",
    3: "retrieve",
    4: "rerank",
    5: "score",
}


@dataclass(frozen=True)
class RankingPipelineConfig:
    stage0_path: Path = RUNTIME_STAGE0_DIR
    stage1_path: Path = RUNTIME_STAGE1_DIR
    stage2_output_dir: Path = RUNTIME_STAGE2_DIR
    stage3_output_dir: Path = RUNTIME_STAGE3_DIR
    stage4_output_dir: Path = RUNTIME_STAGE4_DIR
    stage5_output_dir: Path = RUNTIME_STAGE5_DIR
    candidates_path: Path = CANDIDATES_JSONL_PATH
    config_path: Path = ROOT_DIR / "config.yaml"
    random_seed: int = STAGE1_RANDOM_SEED
    print_summaries: bool = True


@dataclass(frozen=True)
class StageTiming:
    stage: int
    label: str
    elapsed_seconds: float


@dataclass(frozen=True)
class RankingPipelineResult:
    stage1: Stage1RunResult
    stage2: Stage2Result
    stage3: Stage3Result
    stage4: Stage4Result
    stage5: Stage5Result
    timings: tuple[StageTiming, ...]
    total_elapsed_seconds: float
    final_csv_path: Path


def print_pipeline_timing(result: RankingPipelineResult) -> None:
    print("\n--- Pipeline timing ---")
    for timing in result.timings:
        label = f"Stage {timing.stage} ({timing.label})"
        print(f"  {label:<22} {timing.elapsed_seconds:>7.2f}s")
    print(f"  {'Total':<22} {result.total_elapsed_seconds:>7.2f}s")


def run_ranking_pipeline(
    config: RankingPipelineConfig | None = None,
    *,
    print_timing: bool = True,
) -> RankingPipelineResult:
    """
    Run stages 1–5 in sequence. Requires Stage 0 precompute artifacts on disk.

    Individual stage runners remain available for standalone use.
    """
    cfg = config or RankingPipelineConfig()
    timings: list[StageTiming] = []
    pipeline_start = perf_counter()

    stage1_start = perf_counter()
    stage1 = run_stage1_filter(
        cfg.stage0_path,
        stage1_path=cfg.stage1_path,
        output_dir=cfg.stage1_path,
        random_seed=cfg.random_seed,
        print_summary=cfg.print_summaries,
    )
    timings.append(
        StageTiming(1, _STAGE_LABELS[1], perf_counter() - stage1_start)
    )

    stage2_start = perf_counter()
    stage2 = run_stage2(
        stage1_path=cfg.stage1_path,
        artifacts_path=cfg.stage0_path,
        candidates_path=cfg.candidates_path,
        output_dir=cfg.stage2_output_dir,
        config_path=cfg.config_path,
    )
    if cfg.print_summaries:
        print_stage2_summary(stage2)
    timings.append(
        StageTiming(2, _STAGE_LABELS[2], perf_counter() - stage2_start)
    )

    stage2_gated = cfg.stage2_output_dir / "stage2_gated.parquet"
    stage3_start = perf_counter()
    stage3 = run_stage3(
        stage2_path=stage2_gated,
        artifacts_path=cfg.stage0_path,
        output_dir=cfg.stage3_output_dir,
        config_path=cfg.config_path,
    )
    if cfg.print_summaries:
        print_stage3_summary(stage3)
    timings.append(
        StageTiming(3, _STAGE_LABELS[3], perf_counter() - stage3_start)
    )

    stage3_retrieved = cfg.stage3_output_dir / "stage3_retrieved.parquet"
    stage4_start = perf_counter()
    stage4 = run_stage4(
        stage3_path=stage3_retrieved,
        output_dir=cfg.stage4_output_dir,
        config_path=cfg.config_path,
        candidates_path=cfg.candidates_path,
    )
    if cfg.print_summaries:
        print_stage4_summary(stage4)
    timings.append(
        StageTiming(4, _STAGE_LABELS[4], perf_counter() - stage4_start)
    )

    stage4_reranked = cfg.stage4_output_dir / "stage4_reranked.parquet"
    stage5_start = perf_counter()
    stage5 = run_stage5(
        stage4_path=stage4_reranked,
        output_dir=cfg.stage5_output_dir,
        config_path=cfg.config_path,
    )
    if cfg.print_summaries:
        print_stage5_summary(stage5)
    timings.append(
        StageTiming(5, _STAGE_LABELS[5], perf_counter() - stage5_start)
    )

    total_elapsed = perf_counter() - pipeline_start
    result = RankingPipelineResult(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
        timings=tuple(timings),
        total_elapsed_seconds=total_elapsed,
        final_csv_path=stage5.csv_path,
    )

    if print_timing:
        print_pipeline_timing(result)

    return result
