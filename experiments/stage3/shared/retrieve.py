"""Stage 3 runner orchestrator — loads precomputed artifacts only (no ONNX)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from experiments.stage3.shared.config_runner import RunnerConfig, load_runner_config
from experiments.stage3.shared.dense_retrieve import (
    build_id_selector,
    dense_retrieve_q1,
    dense_retrieve_q2,
)
from experiments.stage3.shared.fusion import (
    adaptive_cut,
    build_union,
    compute_fused_score,
    compute_q3_penalty,
    compute_rrf,
)
from experiments.stage3.shared.io_precompute import load_manifest, load_query_vectors
from experiments.stage3.shared.io_runner import (
    load_skill_features,
    load_stage2_gated,
    write_stage3_outputs,
)
from experiments.stage3.shared.io_stage0 import load_retrieval_assets, resolve_stage0_dir
from experiments.stage3.shared.skill_retrieve import skill_retrieve_l3


@dataclass(frozen=True)
class Stage3Result:
    input_count: int
    union_size: int
    l1_size: int
    l2_size: int
    l3_size: int
    triple_overlap: int
    output_count: int
    threshold: float
    fused_min: float
    fused_max: float
    fused_mean: float
    fused_std: float
    elapsed_seconds: float
    output_dir: Path
    top_k_df: pl.DataFrame
    distribution_df: pl.DataFrame


def _triple_overlap(l1: pl.DataFrame, l2: pl.DataFrame, l3: pl.DataFrame) -> int:
    if l1.height == 0 or l2.height == 0 or l3.height == 0:
        return 0
    s1 = set(l1["candidate_id"].to_list())
    s2 = set(l2["candidate_id"].to_list())
    s3 = set(l3["candidate_id"].to_list())
    return len(s1 & s2 & s3)


def run(config_path: Path | None = None) -> Stage3Result:
    start = perf_counter()
    config = load_runner_config(config_path)
    manifest = load_manifest(config.precomputed_manifest)

    stage2_df = load_stage2_gated(manifest.cohort_stage2, config)
    skill_features = load_skill_features(manifest.cohort_features)
    survivor_indices = np.load(manifest.survivor_row_indices).astype(np.int64)

    stage0_dir = resolve_stage0_dir(manifest.stage0_pointer)
    assets = load_retrieval_assets(stage0_dir)
    selector = build_id_selector(survivor_indices)

    q1_vec, q2_vec, q3_vec = load_query_vectors(manifest.query_vectors_dir)
    print("Loaded precomputed query vectors (no ONNX)")

    l1 = dense_retrieve_q1(
        assets.index,
        q1_vec,
        config.per_query_k_dense,
        selector,
        assets.row_to_id,
    )
    l2 = dense_retrieve_q2(
        assets.index,
        q2_vec,
        config.per_query_k_dense,
        selector,
        assets.row_to_id,
    )
    l3 = skill_retrieve_l3(skill_features, stage2_df, config)

    union = build_union(l1, l2, l3, config)
    union = compute_rrf(union, config.rrf_k)
    union = compute_q3_penalty(union, assets.vectors, q3_vec, assets.id_to_row)
    union = compute_fused_score(
        union, stage2_df, config.alpha_neg, config.beta_cluster
    )

    top_k, threshold = adaptive_cut(union, config)
    retrieved = top_k.join(stage2_df, on="candidate_id", how="left").sort("stage3_rank")

    rank_df = top_k.select("candidate_id", "stage3_rank")
    distribution = union.join(rank_df, on="candidate_id", how="left").with_columns(
        pl.col("stage3_rank").is_not_null().alias("kept"),
    )

    elapsed = perf_counter() - start
    fused = union["fused_score"]
    summary = {
        "input_count": stage2_df.height,
        "union_size": union.height,
        "l1_size": l1.height,
        "l2_size": l2.height,
        "l3_size": l3.height,
        "triple_overlap": _triple_overlap(l1, l2, l3),
        "threshold": threshold,
        "output_count": retrieved.height,
        "fused_score_min": float(fused.min()),
        "fused_score_max": float(fused.max()),
        "fused_score_mean": float(fused.mean()),
        "fused_score_std": float(fused.std()),
        "elapsed_seconds": round(elapsed, 3),
    }

    write_stage3_outputs(config.output_dir, retrieved, distribution, summary)

    return Stage3Result(
        input_count=stage2_df.height,
        union_size=union.height,
        l1_size=l1.height,
        l2_size=l2.height,
        l3_size=l3.height,
        triple_overlap=summary["triple_overlap"],
        output_count=retrieved.height,
        threshold=threshold,
        fused_min=summary["fused_score_min"],
        fused_max=summary["fused_score_max"],
        fused_mean=summary["fused_score_mean"],
        fused_std=summary["fused_score_std"],
        elapsed_seconds=elapsed,
        output_dir=config.output_dir,
        top_k_df=retrieved,
        distribution_df=distribution,
    )


def print_stage3_summary(result: Stage3Result) -> None:
    print("\n--- Stage 3 summary (runner) ---")
    print(f"Input (Stage 2):  {result.input_count:,}")
    print(f"Union size:       {result.union_size:,}")
    print(f"L1 / L2 / L3:     {result.l1_size:,} / {result.l2_size:,} / {result.l3_size:,}")
    print(f"Triple overlap:   {result.triple_overlap:,}")
    print(
        f"Fused score:      min={result.fused_min:.6f} max={result.fused_max:.6f} "
        f"mean={result.fused_mean:.6f} std={result.fused_std:.6f}"
    )
    print(f"Threshold:        {result.threshold:.6f}")
    print(f"Output:           {result.output_count:,}")
    print(f"Elapsed:          {result.elapsed_seconds:.2f}s")

    df = result.top_k_df
    if df.height == 0:
        return

    display_cols = [
        "candidate_id",
        "stage3_rank",
        "fused_score",
        "q1_score",
        "q2_score",
        "skill_score",
        "q3_neg_sim",
    ]
    present = [c for c in display_cols if c in df.columns]

    print("\n--- Top 10 by fused_score ---")
    for row in df.sort("stage3_rank").head(10).select(present).iter_rows(named=True):
        print("  " + "  ".join(f"{k}={row[k]}" for k in present))

    print("\n--- Bottom 5 in output ---")
    for row in df.sort("stage3_rank", descending=True).head(5).select(present).iter_rows(
        named=True
    ):
        print("  " + "  ".join(f"{k}={row[k]}" for k in present))

    print(f"\nWrote outputs to {result.output_dir}")
