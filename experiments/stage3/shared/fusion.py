"""RRF fusion, Q3 anti-pattern penalty, and adaptive top-k cutoff."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from experiments.stage3.shared.config_runner import RunnerConfig


def build_union(
    l1: pl.DataFrame,
    l2: pl.DataFrame,
    l3: pl.DataFrame,
    config: RunnerConfig,
) -> pl.DataFrame:
    ids: set[str] = set()
    for frame in (l1, l2, l3):
        if frame.height > 0:
            ids.update(frame["candidate_id"].to_list())

    if not ids:
        return pl.DataFrame()

    union = pl.DataFrame({"candidate_id": sorted(ids)})
    union = union.join(l1, on="candidate_id", how="left")
    union = union.join(l2, on="candidate_id", how="left")
    union = union.join(l3, on="candidate_id", how="left")

    union = union.with_columns(
        pl.col("q1_rank").fill_null(config.miss_penalty_dense).cast(pl.Int64),
        pl.col("q2_rank").fill_null(config.miss_penalty_dense).cast(pl.Int64),
        pl.col("skill_rank").fill_null(config.miss_penalty_skill).cast(pl.Int64),
    )
    return union


def compute_rrf(union_df: pl.DataFrame, rrf_k: int) -> pl.DataFrame:
    return union_df.with_columns(
        (
            1.0 / (rrf_k + pl.col("q1_rank"))
            + 1.0 / (rrf_k + pl.col("q2_rank"))
            + 1.0 / (rrf_k + pl.col("skill_rank"))
        ).alias("rrf_score")
    )


def compute_q3_penalty(
    union_df: pl.DataFrame,
    vectors: np.ndarray,
    q3_vec: np.ndarray,
    id_to_row: dict[str, int],
) -> pl.DataFrame:
    candidate_ids = union_df["candidate_id"].to_list()
    row_indices = np.array([id_to_row[cid] for cid in candidate_ids], dtype=np.int64)
    union_vecs = vectors[row_indices]
    q3_sims = union_vecs @ q3_vec.astype(np.float32)
    return union_df.with_columns(pl.Series("q3_neg_sim", q3_sims.astype(np.float64)))


def compute_fused_score(
    union_df: pl.DataFrame,
    stage2_df: pl.DataFrame,
    alpha_neg: float,
    beta_cluster: float,
) -> pl.DataFrame:
    merged = union_df.join(
        stage2_df.select("candidate_id", "dist_to_centroid"),
        on="candidate_id",
        how="left",
    )

    if beta_cluster == 0.0:
        cluster_term = pl.lit(0.0)
    else:
        cluster_term = beta_cluster * (
            1.0 / (1.0 + pl.col("dist_to_centroid").fill_null(float("inf")))
        )

    return merged.with_columns(
        (
            pl.col("rrf_score")
            - alpha_neg * pl.col("q3_neg_sim")
            + cluster_term
        ).alias("fused_score")
    ).drop("dist_to_centroid")


def adaptive_cut(
    union_df: pl.DataFrame,
    config: RunnerConfig,
) -> tuple[pl.DataFrame, float]:
    if union_df.height == 0:
        raise ValueError("Union is empty — cannot produce Stage 3 output")

    fused = union_df["fused_score"]
    mu = float(fused.mean())
    sigma = float(fused.std())
    threshold = mu - config.z_threshold * sigma

    above = union_df.filter(pl.col("fused_score") >= threshold)
    count = above.height

    if count > config.max_k:
        selected = (
            above.sort(["fused_score", "candidate_id"], descending=[True, False])
            .head(config.max_k)
        )
    elif count < config.min_k:
        warnings.warn(
            f"Only {count} candidates above threshold {threshold:.6f}; "
            f"taking top {config.min_k} by fused_score",
            stacklevel=2,
        )
        selected = (
            union_df.sort(["fused_score", "candidate_id"], descending=[True, False])
            .head(config.min_k)
        )
    else:
        selected = above.sort(["fused_score", "candidate_id"], descending=[True, False])

    output_count = selected.height
    if output_count < config.min_k or output_count > config.max_k:
        raise ValueError(
            f"Stage 3 output count {output_count} outside bounds "
            f"[{config.min_k}, {config.max_k}]. threshold={threshold:.6f}, "
            f"union_size={union_df.height}"
        )

    selected = selected.with_columns(
        pl.int_range(1, pl.len() + 1).alias("stage3_rank")
    )
    return selected, threshold
