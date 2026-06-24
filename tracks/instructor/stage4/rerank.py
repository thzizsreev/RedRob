"""Stage 4 cross-encoder reranking orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import polars as pl

from tracks.instructor.stage4.config import Stage4Config, load_stage4_config
from tracks.instructor.stage4.io import (
    load_stage3_retrieved,
    resolve_candidate_texts,
    write_stage4_outputs,
)
from tracks.instructor.stage4.pairs import (
    build_query_passage_pairs,
    prepare_candidate_texts,
    prepare_jd_text,
)
from tracks.instructor.stage4.score import load_cross_encoder, score_pairs


@dataclass(frozen=True)
class Stage4Result:
    input_count: int
    output_count: int
    inference_seconds: float
    elapsed_seconds: float
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    pairs_per_second: float
    output_dir: Path
    reranked_df: pl.DataFrame


def run(
    stage3_path: Path,
    output_dir: Path,
    config_path: Path,
    *,
    candidates_path: Path | None = None,
    features_path: Path | None = None,
) -> Stage4Result:
    start = perf_counter()
    config = load_stage4_config(config_path)
    if candidates_path is not None:
        config = replace(config, candidates_jsonl_path=candidates_path.resolve())
    if features_path is not None:
        config = replace(config, candidate_features_path=features_path.resolve())

    stage3_df = load_stage3_retrieved(stage3_path, config)
    candidate_ids = stage3_df["candidate_id"].cast(pl.Utf8).to_list()

    text_by_id = resolve_candidate_texts(candidate_ids, config)
    encoder = load_cross_encoder(config)
    jd_text = prepare_jd_text(config, encoder.tokenizer)
    candidate_texts = prepare_candidate_texts(
        candidate_ids, text_by_id, config, encoder.tokenizer
    )
    pairs = build_query_passage_pairs(candidate_ids, jd_text, candidate_texts)

    infer_start = perf_counter()
    scores = score_pairs(encoder, pairs, config)
    inference_seconds = perf_counter() - infer_start

    score_values = [scores.get(cid, config.empty_score) for cid in candidate_ids]
    scored = stage3_df.with_columns(
        pl.Series("cross_encoder_score", score_values),
        pl.lit(config.model_id).alias("stage4_model_id"),
    )

    ranked = scored.sort(["cross_encoder_score", "candidate_id"], descending=[True, False])
    ranked = ranked.with_columns(
        pl.int_range(1, ranked.height + 1).alias("stage4_rank"),
    )

    rank_delta = (
        ranked.with_columns(
            (pl.col("stage3_rank") - pl.col("stage4_rank")).abs().alias("rank_delta")
        )
        .filter(pl.col("rank_delta") >= config.rank_delta_threshold)
        .select(
            "candidate_id",
            "stage3_rank",
            "stage4_rank",
            "rank_delta",
            "cross_encoder_score",
            "fused_score",
        )
        .sort("rank_delta", descending=True)
    )

    keep_n = min(config.keep_n, ranked.height)
    reranked = ranked.head(keep_n).sort("stage4_rank")

    ce_scores = reranked["cross_encoder_score"]
    elapsed = perf_counter() - start
    pairs_per_second = len(candidate_ids) / inference_seconds if inference_seconds > 0 else 0.0

    summary = {
        "input_count": stage3_df.height,
        "output_count": reranked.height,
        "model_id": config.model_id,
        "onnx_model_path": str(config.onnx_model_path),
        "batch_size": config.batch_size,
        "inference_seconds": round(inference_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "pairs_per_second": round(pairs_per_second, 2),
        "score_min": float(ce_scores.min()),
        "score_max": float(ce_scores.max()),
        "score_mean": float(ce_scores.mean()),
        "score_std": float(ce_scores.std()) if reranked.height > 1 else 0.0,
        "keep_n": config.keep_n,
    }

    write_stage4_outputs(output_dir, reranked, rank_delta, summary)

    return Stage4Result(
        input_count=stage3_df.height,
        output_count=reranked.height,
        inference_seconds=inference_seconds,
        elapsed_seconds=elapsed,
        score_min=summary["score_min"],
        score_max=summary["score_max"],
        score_mean=summary["score_mean"],
        score_std=summary["score_std"],
        pairs_per_second=pairs_per_second,
        output_dir=output_dir,
        reranked_df=reranked,
    )


def print_stage4_summary(result: Stage4Result) -> None:
    print("\n--- Stage 4 summary ---")
    print(f"Input (Stage 3):   {result.input_count:,}")
    print(f"Output:            {result.output_count:,}")
    print(
        f"Cross-encoder:     min={result.score_min:.4f} max={result.score_max:.4f} "
        f"mean={result.score_mean:.4f} std={result.score_std:.4f}"
    )
    print(f"Inference:         {result.inference_seconds:.2f}s ({result.pairs_per_second:.1f} pairs/s)")
    print(f"Elapsed total:     {result.elapsed_seconds:.2f}s")

    df = result.reranked_df
    if df.height == 0:
        return

    display_cols = [
        "candidate_id",
        "stage4_rank",
        "stage3_rank",
        "cross_encoder_score",
        "fused_score",
    ]
    present = [c for c in display_cols if c in df.columns]

    print("\n--- Top 10 by cross_encoder_score ---")
    for row in df.sort("stage4_rank").head(10).iter_rows(named=True):
        parts = [f"{k}={row[k]}" for k in present]
        print("  " + "  ".join(parts))

    print(f"\nWrote outputs to {result.output_dir}")
