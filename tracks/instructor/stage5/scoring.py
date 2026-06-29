"""Stage 5 v2 distribution-aware cascade scoring."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import polars as pl

from tracks.instructor.stage5.config import Stage5Config
from tracks.instructor.stage5.normalize import min_max_normalize

_STALE_DAYS_DEFAULT = 999


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        parts = str(value).split("-")
        if len(parts) != 3:
            return None
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError):
        return None


def _days_since_active(last_active: str | None, current_date: date) -> int:
    active = _parse_date(last_active)
    if active is None:
        return _STALE_DAYS_DEFAULT
    return max((current_date - active).days, 0)


def _pool_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values))


def _scale_tier(raw: np.ndarray, target_std: float, tier_name: str) -> np.ndarray:
    raw_std = _pool_std(raw)
    if raw_std <= 0:
        warnings.warn(f"{tier_name} raw std is 0; setting scaled column to 0 for all", stacklevel=2)
        return np.zeros_like(raw, dtype=np.float64)
    factor = target_std / raw_std
    return raw * factor


def _classify_avail_unit(
    interview: float | None,
    offer: float | None,
    days_since_active: int,
    config: Stage5Config,
) -> tuple[str, int]:
    avail = config.availability

    tier_c = (
        (interview is not None and interview < avail.tier_c_interview_max)
        or days_since_active > avail.tier_c_recency_min_days
    )
    if tier_c:
        return "C", -1

    tier_a = (
        interview is not None
        and interview >= avail.tier_a_interview_min
        and days_since_active <= avail.tier_a_recency_max_days
    )
    if tier_a and offer is not None and offer >= 0:
        tier_a = offer >= avail.tier_a_offer_min

    if tier_a:
        return "A", 1

    return "B", 0


def _location_unit(tier: str | None, config: Stage5Config) -> int:
    key = str(tier or "unknown")
    return config.logistics.location_units.get(key, config.logistics.location_units.get("unknown", 0))


def _workmode_unit(mode: str | None, config: Stage5Config) -> int:
    if not mode:
        return 0
    if str(mode).lower() in {"hybrid", "flexible"}:
        return config.logistics.workmode_match_unit
    return 0


def _notice_unit(notice_days: int | None, config: Stage5Config) -> int:
    log = config.logistics
    if notice_days is None:
        return 0
    if notice_days <= log.notice_short_max_days:
        return log.notice_short_unit
    if notice_days > log.notice_long_min_days:
        return log.notice_long_unit
    return log.notice_medium_unit


def apply_scoring(df: pl.DataFrame, config: Stage5Config) -> pl.DataFrame:
    borda = config.borda
    cascade = config.cascade
    t2_cfg = config.tier2
    exp = borda.q_amplification_exponent

    work = df.with_columns(
        [
            pl.col("cross_encoder_score").fill_nan(0.0).fill_null(0.0),
            pl.col("q1_score").fill_nan(0.0).fill_null(0.0),
            pl.col("q2_score").fill_nan(0.0).fill_null(0.0),
        ]
    )

    work = work.with_columns(
        [
            pl.col("cross_encoder_score").rank(method="dense", descending=True).alias("rank_ce"),
            pl.col("q1_score").rank(method="dense", descending=True).alias("rank_q1"),
            pl.col("q2_score").rank(method="dense", descending=True).alias("rank_q2"),
        ]
    )

    work = work.with_columns(
        [
            pl.col("rank_q1").cast(pl.Float64).pow(exp).alias("rank_q1_amp"),
            pl.col("rank_q2").cast(pl.Float64).pow(exp).alias("rank_q2_amp"),
        ]
    )

    work = work.with_columns(
        (
            borda.w_ce * pl.col("rank_ce")
            + borda.w_q1 * pl.col("rank_q1_amp")
            + borda.w_q2 * pl.col("rank_q2_amp")
        ).alias("borda_sum")
    )

    borda_norm = min_max_normalize(work["borda_sum"].to_numpy())
    if len(set(work["borda_sum"].to_list())) <= 1:
        warnings.warn(
            "borda_sum is flat across all candidates; setting borda_primary=0.5 for all",
            stacklevel=2,
        )
    work = work.with_columns(pl.Series("borda_primary", 1.0 - borda_norm))

    t1_std = _pool_std(work["borda_primary"].to_numpy())
    work = work.with_columns(pl.lit(t1_std).alias("t1_std"))

    sweet_bonus = (
        pl.when(pl.col("in_sweet_spot") == True)
        .then(pl.lit(t2_cfg.sweet_spot_bonus))
        .otherwise(pl.lit(0.0))
        .alias("sweet_bonus")
    )
    work = work.with_columns(sweet_bonus)

    work = work.with_columns(
        (
            pl.col("sweet_bonus")
            + pl.col("optional_bonus").fill_null(0.0)
            - pl.col("title_chasing_penalty").fill_null(0.0)
            - pl.col("ambiguity_penalty").fill_null(0.0)
            - pl.col("closed_source_penalty").fill_null(0.0)
        ).alias("tier2_raw")
    )

    t2_std = _pool_std(work["tier2_raw"].to_numpy())
    target_t2_std = cascade.tier2_ratio * t1_std
    tier2_scaled = _scale_tier(work["tier2_raw"].to_numpy(), target_t2_std, "tier2")
    work = work.with_columns(
        [
            pl.lit(t2_std).alias("t2_std"),
            pl.lit(target_t2_std).alias("target_t2_std"),
            pl.Series("tier2_scaled", tier2_scaled),
        ]
    )

    days_list: list[int] = []
    avail_tiers: list[str] = []
    avail_units: list[int] = []
    for row in work.iter_rows(named=True):
        days = _days_since_active(row.get("last_active_date"), config.current_date)
        days_list.append(days)
        interview = row.get("interview_completion_rate")
        offer = row.get("offer_acceptance_rate")
        if interview is not None:
            interview = float(interview)
        if offer is not None:
            offer = float(offer)
        tier, unit = _classify_avail_unit(interview, offer, days, config)
        avail_tiers.append(tier)
        avail_units.append(unit)

    work = work.with_columns(
        [
            pl.Series("days_since_active", days_list),
            pl.Series("avail_tier", avail_tiers),
            pl.Series("avail_unit", avail_units),
        ]
    )

    t3_std = _pool_std(np.array(avail_units, dtype=np.float64))
    target_t3_std = cascade.tier3_ratio * t1_std
    tier3_scaled = _scale_tier(np.array(avail_units, dtype=np.float64), target_t3_std, "tier3")
    work = work.with_columns(
        [
            pl.lit(t3_std).alias("t3_std"),
            pl.lit(target_t3_std).alias("target_t3_std"),
            pl.Series("tier3_scaled", tier3_scaled),
        ]
    )

    location_units = [_location_unit(row.get("location_tier"), config) for row in work.iter_rows(named=True)]
    workmode_units = [_workmode_unit(row.get("preferred_work_mode"), config) for row in work.iter_rows(named=True)]
    notice_units = [_notice_unit(row.get("notice_period_days"), config) for row in work.iter_rows(named=True)]

    work = work.with_columns(
        [
            pl.Series("location_unit", location_units),
            pl.Series("workmode_unit", workmode_units),
            pl.Series("notice_unit", notice_units),
        ]
    )

    work = work.with_columns(
        (pl.col("location_unit") + pl.col("workmode_unit") + pl.col("notice_unit")).alias("tier4_raw")
    )

    t4_std = _pool_std(work["tier4_raw"].to_numpy())
    target_t4_std = cascade.tier4_ratio * t1_std
    tier4_scaled = _scale_tier(work["tier4_raw"].to_numpy(), target_t4_std, "tier4")
    work = work.with_columns(
        [
            pl.lit(t4_std).alias("t4_std"),
            pl.lit(target_t4_std).alias("target_t4_std"),
            pl.Series("tier4_scaled", tier4_scaled),
        ]
    )

    work = work.with_columns(
        (
            pl.col("borda_primary")
            + pl.col("tier2_scaled")
            + pl.col("tier3_scaled")
            + pl.col("tier4_scaled")
        ).alias("final_score")
    )

    return work
