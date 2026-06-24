"""Seven-layer composite scoring for Stage 5."""

from __future__ import annotations

import numpy as np
import polars as pl

from tracks.instructor.stage5.config import Stage5Config
from tracks.instructor.stage5.must_have import (
    assessment_coverage,
    keyword_ratio,
    must_have_floor_multiplier,
)
from tracks.instructor.stage5.normalize import min_max_normalize
from tracks.instructor.stage5.signals import (
    clamp,
    days_inactive,
    interview_factor,
    market_factor,
    offer_factor,
    open_factor,
    recency_factor,
    resp_factor,
    speed_factor,
)


def _optional_bonus_count(
    skills: list[dict] | None,
    has_github: bool,
    external_validation_score: float,
    config: Stage5Config,
) -> int:
    names = [str(s.get("name", "")).lower() for s in (skills or []) if s.get("name")]
    if not names:
        return 0
    text = " ".join(names)
    count = 0
    validation_floor = config.penalties.validation_floor
    for category, keywords in config.optional_bonus.categories.items():
        if not any(kw in text for kw in keywords):
            continue
        if category == "oss":
            if has_github and external_validation_score >= validation_floor:
                count += 1
        else:
            count += 1
    return count


def _location_adj(tier: str | None, config: Stage5Config) -> float:
    lookup = {
        "preferred": config.logistics.loc_preferred_bonus,
        "acceptable": config.logistics.loc_acceptable_bonus,
        "unknown": 0.0,
        "outside_india": -config.logistics.loc_outside_penalty,
    }
    return lookup.get(str(tier or "unknown"), 0.0)


def _workmode_adj(mode: str | None, config: Stage5Config) -> float:
    if not mode:
        return 0.0
    mode = str(mode).lower()
    if mode == "remote":
        return -config.logistics.workmode_remote_penalty
    if mode in ("hybrid", "flexible", "onsite"):
        return config.logistics.workmode_fit_bonus
    return 0.0


def _notice_adj(notice_days: int | None, config: Stage5Config) -> float:
    if notice_days is None or notice_days <= 30:
        return 0.0
    excess = notice_days - 30
    penalty = min(
        config.logistics.notice_penalty_cap,
        (excess / 90.0) * config.logistics.notice_penalty_scale,
    )
    return -penalty


def apply_scoring(df: pl.DataFrame, config: Stage5Config) -> pl.DataFrame:
    cw = config.core_weights
    cs = config.career_shape
    pen = config.penalties
    avail_cfg = config.availability
    opt_cfg = config.optional_bonus

    score_cols = ["cross_encoder_score", "fused_score", "q1_score", "q2_score", "q3_neg_sim"]
    work = df.with_columns(
        [pl.col(c).fill_nan(0.0).fill_null(0.0).alias(c) for c in score_cols]
    )

    ce_norm = min_max_normalize(work["cross_encoder_score"].to_numpy())
    fused_norm = min_max_normalize(work["fused_score"].to_numpy())
    q1_norm = min_max_normalize(work["q1_score"].to_numpy())
    q2_norm = min_max_normalize(work["q2_score"].to_numpy())
    q3_norm = min_max_normalize(work["q3_neg_sim"].to_numpy())

    core = (
        cw.w_cross * ce_norm
        + cw.w_fused * fused_norm
        + cw.w_q1 * q1_norm
        + cw.w_q2 * q2_norm
    )

    n = df.height
    keyword_cov = np.zeros(n)
    assessment_cov = np.zeros(n)
    combined_cov = np.zeros(n)
    floor_mult = np.zeros(n)
    core_floored = np.zeros(n)
    shape_mult = np.zeros(n)
    shaped = np.zeros(n)
    title_chasing = np.zeros(n)
    q3_residual = np.zeros(n)
    closed_source = np.zeros(n)
    ambiguity = np.zeros(n)
    consulting_resid = np.zeros(n)
    total_penalty = np.zeros(n)
    penalized = np.zeros(n)
    optional_bonus = np.zeros(n)
    bonused = np.zeros(n)
    availability_multiplier = np.zeros(n)
    availability_adj = np.zeros(n)
    location_adj = np.zeros(n)
    workmode_adj = np.zeros(n)
    notice_adj = np.zeros(n)
    logistics_adjustment = np.zeros(n)
    final_score = np.zeros(n)

    rows = work.to_dicts()
    for i, row in enumerate(rows):
        skills = row.get("skills") or []
        assessments = row.get("skill_assessment_scores") or {}
        sem = float(q1_norm[i])

        kw = keyword_ratio(skills, config)
        ac = assessment_coverage(assessments, skills, config, sem)
        combined, mult = must_have_floor_multiplier(kw, sem, ac, config.must_have_floor_min)

        keyword_cov[i] = kw
        assessment_cov[i] = ac
        combined_cov[i] = combined
        floor_mult[i] = mult
        core_floored[i] = core[i] * mult

        product_frac = float(row.get("product_company_fraction") or 0.0)
        sm = cs.product_floor + (1.0 - cs.product_floor) * product_frac
        if row.get("in_sweet_spot"):
            sm *= cs.sweet_spot_bonus
        elif row.get("exp_band") == "near_band":
            sm *= cs.near_band_factor
        if row.get("stale_coding"):
            sm *= cs.stale_coding_factor
        if not row.get("has_any_production_role"):
            sm *= cs.no_production_factor
        shape_mult[i] = sm
        shaped[i] = core_floored[i] * sm

        hops = int(row.get("short_hop_count") or 0)
        tc = min(pen.short_hop_penalty_cap, hops * pen.per_hop_penalty)
        q3r = pen.q3_penalty_weight * float(q3_norm[i])
        ext_val = float(row.get("external_validation_score") or 0.0)
        has_gh = bool(row.get("has_github"))
        cs_pen = (
            pen.closed_source_penalty_value
            if ext_val < pen.validation_floor and not has_gh
            else 0.0
        )
        title_amb = 1.0 if row.get("title_ambiguous") else 0.0
        near = 1.0 if row.get("exp_band") == "near_band" else 0.0
        amb = pen.title_ambiguous_penalty * title_amb + pen.near_band_penalty * near
        cons = (
            pen.consulting_heavy_penalty
            if row.get("career_type") == "consulting_heavy"
            else 0.0
        )
        total = tc + q3r + cs_pen + amb + cons

        title_chasing[i] = tc
        q3_residual[i] = q3r
        closed_source[i] = cs_pen
        ambiguity[i] = amb
        consulting_resid[i] = cons
        total_penalty[i] = total
        penalized[i] = shaped[i] - total

        opt_count = _optional_bonus_count(skills, has_gh, ext_val, config)
        ob = min(opt_cfg.optional_bonus_cap, opt_count * opt_cfg.per_category_bonus)
        optional_bonus[i] = ob
        bonused[i] = penalized[i] + ob

        inactive_days = days_inactive(row.get("last_active_date"), config.current_date)
        rf = resp_factor(
            row.get("recruiter_response_rate"),
            avail_cfg.good_response_rate,
            avail_cfg.response_floor,
        )
        sf = speed_factor(
            row.get("avg_response_time_hours"),
            avail_cfg.slow_response_hours,
            avail_cfg.response_decay_window_hours,
            avail_cfg.speed_floor,
        )
        rcf = recency_factor(
            inactive_days,
            avail_cfg.fresh_days,
            avail_cfg.recency_decay_window,
            avail_cfg.recency_floor,
        )
        of = open_factor(row.get("open_to_work_flag"), avail_cfg.not_open_factor)
        iff = interview_factor(row.get("interview_completion_rate"), avail_cfg.interview_floor)
        off = offer_factor(row.get("offer_acceptance_rate"), avail_cfg.offer_floor)
        mf = market_factor(
            row.get("applications_submitted_30d"),
            row.get("open_to_work_flag"),
            avail_cfg.market_inactive_factor,
        )
        am = clamp(rf * sf * rcf * of * iff * off * mf, avail_cfg.avail_min, 1.0)
        availability_multiplier[i] = am
        availability_adj[i] = bonused[i] * am

        loc = _location_adj(row.get("location_tier"), config)
        wm = _workmode_adj(row.get("preferred_work_mode"), config)
        na = _notice_adj(row.get("notice_period_days"), config)
        location_adj[i] = loc
        workmode_adj[i] = wm
        notice_adj[i] = na
        logistics_adjustment[i] = loc + wm + na
        final_score[i] = availability_adj[i] + logistics_adjustment[i]

    return work.with_columns(
        pl.Series("ce_norm", ce_norm),
        pl.Series("fused_norm", fused_norm),
        pl.Series("q1_norm", q1_norm),
        pl.Series("q2_norm", q2_norm),
        pl.Series("q3_norm", q3_norm),
        pl.Series("core", core),
        pl.Series("keyword_ratio", keyword_cov),
        pl.Series("assessment_cov", assessment_cov),
        pl.Series("combined_coverage", combined_cov),
        pl.Series("must_have_floor_multiplier", floor_mult),
        pl.Series("core_floored", core_floored),
        pl.Series("shape_mult", shape_mult),
        pl.Series("shaped", shaped),
        pl.Series("title_chasing_penalty", title_chasing),
        pl.Series("q3_residual_penalty", q3_residual),
        pl.Series("closed_source_penalty", closed_source),
        pl.Series("ambiguity_penalty", ambiguity),
        pl.Series("consulting_resid_penalty", consulting_resid),
        pl.Series("total_penalty", total_penalty),
        pl.Series("penalized", penalized),
        pl.Series("optional_bonus", optional_bonus),
        pl.Series("bonused", bonused),
        pl.Series("availability_multiplier", availability_multiplier),
        pl.Series("availability_adj", availability_adj),
        pl.Series("location_adj", location_adj),
        pl.Series("workmode_adj", workmode_adj),
        pl.Series("notice_adj", notice_adj),
        pl.Series("logistics_adjustment", logistics_adjustment),
        pl.Series("final_score", final_score),
    )
