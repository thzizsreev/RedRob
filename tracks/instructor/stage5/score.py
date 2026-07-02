"""Stage 5 composite scoring orchestrator."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import polars as pl

from tracks.instructor.stage5.config import Stage5Config, load_stage5_config
from tracks.instructor.stage5.io import (
    join_scoring_inputs,
    load_stage4_reranked,
    warn_input_count,
    write_stage5_outputs,
)
from tracks.instructor.stage5.reasoning import compose_reasoning
from tracks.instructor.stage5.scoring import apply_scoring
from tracks.instructor.stage5.validate import validate_ranking_csv, validate_submission_csv


@dataclass(frozen=True)
class Stage5Result:
    input_count: int
    output_count: int
    score_min: float
    score_max: float
    score_mean: float
    elapsed_seconds: float
    csv_path: Path
    output_dir: Path
    summary: dict = field(default_factory=dict)
    top_df: pl.DataFrame | None = None


def _enforce_monotonic_scores(scores: list[float]) -> tuple[list[float], int]:
    if not scores:
        return scores, 0
    out = [scores[0]]
    corrections = 0
    for i in range(1, len(scores)):
        prev = out[i - 1]
        cur = scores[i]
        if cur > prev:
            corrections += 1
            out.append(prev)
        else:
            out.append(cur)
    return out, corrections


def _rescale_factor(target: float, raw_std: float) -> float:
    if raw_std <= 0:
        return 0.0
    return target / raw_std


def _build_summary(
    scored: pl.DataFrame,
    top: pl.DataFrame,
    stage4_df: pl.DataFrame,
    config: Stage5Config,
    top_n: int,
    elapsed: float,
    monotonic_corrections: int,
) -> dict:
    t1_std = float(scored["t1_std"][0])
    t2_std = float(scored["t2_std"][0])
    t3_std = float(scored["t3_std"][0])
    t4_std = float(scored["t4_std"][0])
    target_t2 = float(scored["target_t2_std"][0])
    target_t3 = float(scored["target_t3_std"][0])
    target_t4 = float(scored["target_t4_std"][0])

    tier_counts = scored.group_by("avail_tier").len().sort("avail_tier")
    tier_a = tier_b = tier_c = 0
    for row in tier_counts.iter_rows(named=True):
        label = str(row["avail_tier"])
        count = int(row["len"])
        if label == "A":
            tier_a = count
        elif label == "B":
            tier_b = count
        elif label == "C":
            tier_c = count

    borda = scored["borda_primary"]
    tier2 = scored["tier2_scaled"]
    final = scored["final_score"]

    return {
        "input_count": stage4_df.height,
        "output_count": top_n,
        "team_id": config.team_id,
        "score_min": float(top["final_score"].min()) if top_n else 0.0,
        "score_max": float(top["final_score"].max()) if top_n else 0.0,
        "score_mean": float(top["final_score"].mean()) if top_n else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "monotonic_corrections": monotonic_corrections,
        "t1_std": t1_std,
        "t2_std": t2_std,
        "t3_std": t3_std,
        "t4_std": t4_std,
        "target_t2_std": target_t2,
        "target_t3_std": target_t3,
        "target_t4_std": target_t4,
        "tier2_rescale_factor": _rescale_factor(target_t2, t2_std),
        "tier3_rescale_factor": _rescale_factor(target_t3, t3_std),
        "tier4_rescale_factor": _rescale_factor(target_t4, t4_std),
        "borda_primary_min": float(borda.min()),
        "borda_primary_max": float(borda.max()),
        "borda_primary_mean": float(borda.mean()),
        "borda_primary_std": float(borda.std()) if scored.height > 1 else 0.0,
        "tier2_scaled_min": float(tier2.min()),
        "tier2_scaled_max": float(tier2.max()),
        "final_score_min": float(final.min()),
        "final_score_max": float(final.max()),
        "final_score_mean": float(final.mean()),
        "final_score_std": float(final.std()) if scored.height > 1 else 0.0,
        "avail_tier_a": tier_a,
        "avail_tier_b": tier_b,
        "avail_tier_c": tier_c,
        "borda_w_ce": config.borda.w_ce,
        "borda_w_q1": config.borda.w_q1,
        "borda_w_q2": config.borda.w_q2,
        "borda_amp_exp": config.borda.q_amplification_exponent,
        "cascade_tier2_ratio": config.cascade.tier2_ratio,
        "cascade_tier3_ratio": config.cascade.tier3_ratio,
        "cascade_tier4_ratio": config.cascade.tier4_ratio,
    }


def run(
    *,
    stage4_path: Path,
    output_dir: Path,
    config_path: Path,
    include_reasoning: bool = True,
) -> Stage5Result:
    start = perf_counter()
    config = load_stage5_config(config_path)

    stage4_df = load_stage4_reranked(stage4_path)
    input_ids = set(stage4_df["candidate_id"].cast(pl.Utf8).to_list())
    warn_input_count(stage4_df.height)

    joined = join_scoring_inputs(stage4_df, config)
    scored = apply_scoring(joined, config)

    if include_reasoning:
        reasoning_c1: list[str] = []
        reasoning_c2: list[str] = []
        reasoning_c3: list[str | None] = []
        for row in scored.iter_rows(named=True):
            row_dict = dict(row)
            row_dict["chase_pen"] = row_dict.get("title_chasing_penalty")
            row_dict["closed_pen"] = row_dict.get("closed_source_penalty")
            c1, c2, c3, text = compose_reasoning(row_dict)
            reasoning_c1.append(c1)
            reasoning_c2.append(c2)
            reasoning_c3.append(c3)

        scored = scored.with_columns(
            [
                pl.Series("reasoning_clause_1", reasoning_c1),
                pl.Series("reasoning_clause_2", reasoning_c2),
                pl.Series("reasoning_clause_3", [c or "" for c in reasoning_c3]),
            ]
        )

    ranked = scored.sort(["final_score", "candidate_id"], descending=[True, False])
    top_n = min(config.top_n, ranked.height)
    top_ids = set(ranked.head(top_n)["candidate_id"].cast(pl.Utf8).to_list())

    scored_with_flag = ranked.with_columns(
        pl.col("candidate_id").is_in(list(top_ids)).alias("in_top_100")
    )
    top = scored_with_flag.head(top_n).with_columns(
        pl.int_range(1, top_n + 1).alias("rank"),
    )

    raw_scores = [float(s) for s in top["final_score"].to_list()]
    clamped_scores, monotonic_corrections = _enforce_monotonic_scores(raw_scores)
    if monotonic_corrections:
        warnings.warn(
            f"Monotonicity enforcement corrected {monotonic_corrections} score position(s)",
            stacklevel=2,
        )

    submission_rows: list[dict] = []
    for row, score in zip(top.iter_rows(named=True), clamped_scores):
        row_dict = dict(row)
        entry = {
            "candidate_id": row["candidate_id"],
            "rank": row["rank"],
            "score": round(score, 6),
        }
        if include_reasoning:
            row_dict["chase_pen"] = row_dict.get("title_chasing_penalty")
            row_dict["closed_pen"] = row_dict.get("closed_source_penalty")
            _, _, _, reasoning = compose_reasoning(row_dict)
            entry["reasoning"] = reasoning
        submission_rows.append(entry)

    elapsed = perf_counter() - start
    summary = _build_summary(
        scored_with_flag,
        top,
        stage4_df,
        config,
        top_n,
        elapsed,
        monotonic_corrections,
    )

    csv_path = write_stage5_outputs(
        output_dir,
        config.team_id,
        scored_with_flag,
        top,
        submission_rows,
        summary,
        include_reasoning=include_reasoning,
    )
    if include_reasoning:
        validate_submission_csv(
            csv_path,
            expected_rows=top_n,
            input_candidate_ids=input_ids,
        )
    else:
        validate_ranking_csv(
            csv_path,
            expected_rows=top_n,
            input_candidate_ids=input_ids,
        )

    return Stage5Result(
        input_count=stage4_df.height,
        output_count=top_n,
        score_min=summary["score_min"],
        score_max=summary["score_max"],
        score_mean=summary["score_mean"],
        elapsed_seconds=elapsed,
        csv_path=csv_path,
        output_dir=output_dir,
        summary=summary,
        top_df=top,
    )


def print_stage5_summary(result: Stage5Result) -> None:
    s = result.summary
    print("\n=== STAGE 5 COMPLETE ===")
    print(f"Input candidates:   {result.input_count}")
    print(f"t1_std (anchor):    {s.get('t1_std', 0):.6f}")

    print("\n--- Tier 1 (Borda) ---")
    print(
        f"borda_primary:  min={s.get('borda_primary_min', 0):.4f}  "
        f"max={s.get('borda_primary_max', 0):.4f}  "
        f"mean={s.get('borda_primary_mean', 0):.4f}  "
        f"std={s.get('borda_primary_std', 0):.4f}"
    )
    print(
        f"Weights used:   ce={s.get('borda_w_ce')}  q1={s.get('borda_w_q1')}  "
        f"q2={s.get('borda_w_q2')}  amplification_exp={s.get('borda_amp_exp')}"
    )

    print("\n--- Tier 2 (Career shape + penalties) ---")
    print(f"tier2_raw:      std={s.get('t2_std', 0):.6f}")
    print(
        f"target_t2_std:  {s.get('target_t2_std', 0):.6f}  "
        f"(ratio={s.get('cascade_tier2_ratio')})"
    )
    print(f"rescale_factor: {s.get('tier2_rescale_factor', 0):.4f}")
    print(
        f"tier2_scaled:   min={s.get('tier2_scaled_min', 0):.4f}  "
        f"max={s.get('tier2_scaled_max', 0):.4f}"
    )

    print("\n--- Tier 3 (Availability) ---")
    print(f"Tier A:  {s.get('avail_tier_a', 0)} candidates  avail_unit=+1")
    print(f"Tier B:  {s.get('avail_tier_b', 0)} candidates  avail_unit= 0")
    print(f"Tier C:  {s.get('avail_tier_c', 0)} candidates  avail_unit=-1")
    print(
        f"t3_std: {s.get('t3_std', 0):.6f}   target: {s.get('target_t3_std', 0):.6f}   "
        f"rescale: {s.get('tier3_rescale_factor', 0):.4f}"
    )

    print("\n--- Tier 4 (Logistics) ---")
    print(f"tier4_raw:      std={s.get('t4_std', 0):.6f}")
    print(
        f"target_t4_std:  {s.get('target_t4_std', 0):.6f}  "
        f"(ratio={s.get('cascade_tier4_ratio')})"
    )
    print(f"rescale_factor: {s.get('tier4_rescale_factor', 0):.4f}")

    print("\n--- Final score ---")
    print(
        f"final_score:  min={s.get('final_score_min', 0):.4f}  "
        f"max={s.get('final_score_max', 0):.4f}  "
        f"mean={s.get('final_score_mean', 0):.4f}  "
        f"std={s.get('final_score_std', 0):.4f}"
    )

    if result.top_df is not None and result.top_df.height > 0:
        print("\n--- Top 10 ---")
        print(
            "rank | candidate_id   | final_score | borda_primary | tier2_scaled | "
            "avail_tier | location_tier"
        )
        for row in result.top_df.head(10).iter_rows(named=True):
            print(
                f"{row['rank']:>4} | {row['candidate_id']}   | "
                f"{row['final_score']:.4f}      | {row['borda_primary']:.4f}        | "
                f"{row.get('tier2_scaled', 0):.4f}       | "
                f"{row.get('avail_tier', 'B')}          | {row.get('location_tier', 'unknown')}"
            )

    print("\n--- Signals removed vs old formula (confirmed by diagnostics) ---")
    print("product_company_fraction:  NOT used (IQR=0.0, near-constant in pool)")
    print("has_any_production_role:   NOT used (95.3% triggered, near-constant)")
    print("q3_residual_penalty:       NOT used (std=0.0075, near-flat, double-penalizes)")
    print("fused_norm:                NOT used (amplified noise, raw std=0.0076)")
    print("stale_coding:              NOT used (0 triggers)")
    print("consulting_heavy:          NOT used (0 triggers)")
    print("market_factor:             NOT used (near-constant, 290/300 at ceiling)")
    print("resp_factor:               NOT used (215/300 at ceiling)")
    print("Multiplicative availability stack: REMOVED (caused 74% floor collapse)")

    print(f"\nOutput written: {result.csv_path}")
    print(f"Debug written:  {result.output_dir / 'stage5_full_scores.parquet'}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")
