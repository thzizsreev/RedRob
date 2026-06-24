"""Stage 3 multi-query hybrid retrieval orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import polars as pl

from tracks.instructor.core.onnx_embedder import load_embedder, unload_embedder
from tracks.instructor.stage3.config import Stage3Config, load_stage3_config
from tracks.instructor.stage3.dense_retrieve import (
    build_id_selector,
    dense_retrieve_q1,
    dense_retrieve_q2,
)
from tracks.instructor.stage3.fusion import (
    adaptive_cut,
    build_union,
    compute_fused_score,
    compute_q3_penalty,
    compute_rrf,
)
from tracks.instructor.stage3.io import (
    RetrievalAssets,
    build_survivor_row_indices,
    load_retrieval_assets,
    load_stage2_gated,
    write_stage3_outputs,
)
from tracks.instructor.stage3.query_encode import encode_stage3_queries
from tracks.instructor.stage3.sparse_retrieve import bm25_retrieve_q4


@dataclass(frozen=True)
class Stage3Result:
    input_count: int
    union_size: int
    l1_size: int
    l2_size: int
    l4_size: int
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


def _triple_overlap(l1: pl.DataFrame, l2: pl.DataFrame, l4: pl.DataFrame) -> int:
    if l1.height == 0 or l2.height == 0 or l4.height == 0:
        return 0
    s1 = set(l1["candidate_id"].to_list())
    s2 = set(l2["candidate_id"].to_list())
    s4 = set(l4["candidate_id"].to_list())
    return len(s1 & s2 & s4)


def run(
    stage2_path: Path,
    artifacts_path: Path,
    output_dir: Path,
    config_path: Path,
) -> Stage3Result:
    start = perf_counter()
    config = load_stage3_config(config_path)

    stage2_df = load_stage2_gated(stage2_path, config)
    assets = load_retrieval_assets(artifacts_path)
    survivor_indices = build_survivor_row_indices(stage2_df, assets.id_to_row)
    selector = build_id_selector(survivor_indices)

    model = load_embedder()
    try:
        q1_vec, q2_vec, q3_vec = encode_stage3_queries(model, config)
    finally:
        unload_embedder(model)

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
    l4 = bm25_retrieve_q4(
        assets.bm25,
        config.q4_tokens,
        survivor_indices,
        assets.row_to_id,
        config.per_query_k_sparse,
    )

    union = build_union(l1, l2, l4, config)
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
        "l4_size": l4.height,
        "triple_overlap": _triple_overlap(l1, l2, l4),
        "threshold": threshold,
        "output_count": retrieved.height,
        "fused_score_min": float(fused.min()),
        "fused_score_max": float(fused.max()),
        "fused_score_mean": float(fused.mean()),
        "fused_score_std": float(fused.std()),
        "elapsed_seconds": round(elapsed, 3),
    }

    write_stage3_outputs(output_dir, retrieved, distribution, summary)

    return Stage3Result(
        input_count=stage2_df.height,
        union_size=union.height,
        l1_size=l1.height,
        l2_size=l2.height,
        l4_size=l4.height,
        triple_overlap=summary["triple_overlap"],
        output_count=retrieved.height,
        threshold=threshold,
        fused_min=summary["fused_score_min"],
        fused_max=summary["fused_score_max"],
        fused_mean=summary["fused_score_mean"],
        fused_std=summary["fused_score_std"],
        elapsed_seconds=elapsed,
        output_dir=output_dir,
        top_k_df=retrieved,
        distribution_df=distribution,
    )


def print_stage3_summary(result: Stage3Result) -> None:
    print("\n--- Stage 3 summary ---")
    print(f"Input (Stage 2):  {result.input_count:,}")
    print(f"Union size:       {result.union_size:,}")
    print(f"L1 / L2 / L4:     {result.l1_size:,} / {result.l2_size:,} / {result.l4_size:,}")
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
        "bm25_score",
        "q3_neg_sim",
    ]
    present = [c for c in display_cols if c in df.columns]

    print("\n--- Top 10 by fused_score ---")
    top10 = df.sort("stage3_rank").head(10).select(present)
    for row in top10.iter_rows(named=True):
        parts = [f"{k}={row[k]}" for k in present]
        print("  " + "  ".join(parts))

    print("\n--- Bottom 5 in output ---")
    bottom5 = df.sort("stage3_rank", descending=True).head(5).select(present)
    for row in bottom5.iter_rows(named=True):
        parts = [f"{k}={row[k]}" for k in present]
        print("  " + "  ".join(parts))

    print(f"\nWrote outputs to {result.output_dir}")
