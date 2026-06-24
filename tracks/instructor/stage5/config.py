"""Load Stage 5 configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from tracks.shared.paths import CANDIDATES_JSONL_PATH, ROOT_DIR, RUNTIME_STAGE4_DIR, RUNTIME_STAGE5_DIR


@dataclass(frozen=True)
class CoreWeights:
    w_cross: float
    w_fused: float
    w_q1: float
    w_q2: float


@dataclass(frozen=True)
class CareerShapeWeights:
    product_floor: float
    sweet_spot_bonus: float
    near_band_factor: float
    stale_coding_factor: float
    no_production_factor: float


@dataclass(frozen=True)
class PenaltyConfig:
    per_hop_penalty: float
    short_hop_penalty_cap: float
    q3_penalty_weight: float
    validation_floor: float
    closed_source_penalty_value: float
    title_ambiguous_penalty: float
    near_band_penalty: float
    consulting_heavy_penalty: float


@dataclass(frozen=True)
class OptionalBonusConfig:
    per_category_bonus: float
    optional_bonus_cap: float
    categories: dict[str, list[str]]


@dataclass(frozen=True)
class AvailabilityConfig:
    good_response_rate: float
    response_floor: float
    slow_response_hours: float
    response_decay_window_hours: float
    speed_floor: float
    fresh_days: int
    recency_decay_window: int
    recency_floor: float
    not_open_factor: float
    interview_floor: float
    offer_floor: float
    market_inactive_factor: float
    avail_min: float


@dataclass(frozen=True)
class LogisticsConfig:
    loc_preferred_bonus: float
    loc_acceptable_bonus: float
    loc_outside_penalty: float
    workmode_fit_bonus: float
    workmode_remote_penalty: float
    notice_penalty_scale: float
    notice_penalty_cap: float


@dataclass(frozen=True)
class Stage5Config:
    team_id: str
    top_n: int
    current_date: date
    stage4_input_path: Path
    candidates_jsonl_path: Path
    candidate_features_path: Path
    output_dir: Path
    core_weights: CoreWeights
    must_have_floor_min: float
    must_have_keywords: dict[str, list[str]]
    career_shape: CareerShapeWeights
    penalties: PenaltyConfig
    optional_bonus: OptionalBonusConfig
    availability: AvailabilityConfig
    logistics: LogisticsConfig
    enable_popularity_tiebreak: bool
    popularity_tiebreak_cap: float


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _parse_date(value: str) -> date:
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def _lower_keyword_map(raw: dict) -> dict[str, list[str]]:
    return {str(k): [str(v).lower() for v in vals] for k, vals in raw.items()}


def load_stage5_config(config_path: Path) -> Stage5Config:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "stage5" not in raw:
        raise ValueError(f"Missing 'stage5' namespace in {config_path}")

    s5 = raw["stage5"]
    cw = s5["core_weights"]
    core = CoreWeights(
        w_cross=float(cw["w_cross"]),
        w_fused=float(cw["w_fused"]),
        w_q1=float(cw["w_q1"]),
        w_q2=float(cw["w_q2"]),
    )
    weight_sum = core.w_cross + core.w_fused + core.w_q1 + core.w_q2
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"stage5.core_weights must sum to 1.0, got {weight_sum}")

    cs = s5["career_shape"]
    pen = s5["penalties"]
    opt = s5["optional_bonus"]
    avail = s5["availability"]
    log = s5["logistics"]

    return Stage5Config(
        team_id=str(s5["team_id"]),
        top_n=int(s5.get("top_n", 100)),
        current_date=_parse_date(str(s5["current_date"])),
        stage4_input_path=_resolve_path(
            str(s5.get("stage4_input_path", RUNTIME_STAGE4_DIR / "stage4_reranked.parquet"))
        ),
        candidates_jsonl_path=_resolve_path(
            str(s5.get("candidates_jsonl_path", CANDIDATES_JSONL_PATH))
        ),
        candidate_features_path=_resolve_path(
            str(s5.get("candidate_features_path", "artifacts/precomputed/candidate_features.parquet"))
        ),
        output_dir=_resolve_path(str(s5.get("output_dir", RUNTIME_STAGE5_DIR))),
        core_weights=core,
        must_have_floor_min=float(s5.get("must_have_floor_min", 0.4)),
        must_have_keywords=_lower_keyword_map(s5.get("must_have_keywords", {})),
        career_shape=CareerShapeWeights(
            product_floor=float(cs["product_floor"]),
            sweet_spot_bonus=float(cs["sweet_spot_bonus"]),
            near_band_factor=float(cs["near_band_factor"]),
            stale_coding_factor=float(cs["stale_coding_factor"]),
            no_production_factor=float(cs["no_production_factor"]),
        ),
        penalties=PenaltyConfig(
            per_hop_penalty=float(pen["per_hop_penalty"]),
            short_hop_penalty_cap=float(pen["short_hop_penalty_cap"]),
            q3_penalty_weight=float(pen["q3_penalty_weight"]),
            validation_floor=float(pen["validation_floor"]),
            closed_source_penalty_value=float(pen["closed_source_penalty_value"]),
            title_ambiguous_penalty=float(pen["title_ambiguous_penalty"]),
            near_band_penalty=float(pen["near_band_penalty"]),
            consulting_heavy_penalty=float(pen["consulting_heavy_penalty"]),
        ),
        optional_bonus=OptionalBonusConfig(
            per_category_bonus=float(opt["per_category_bonus"]),
            optional_bonus_cap=float(opt["optional_bonus_cap"]),
            categories=_lower_keyword_map(opt.get("categories", {})),
        ),
        availability=AvailabilityConfig(
            good_response_rate=float(avail["good_response_rate"]),
            response_floor=float(avail["response_floor"]),
            slow_response_hours=float(avail["slow_response_hours"]),
            response_decay_window_hours=float(avail["response_decay_window_hours"]),
            speed_floor=float(avail["speed_floor"]),
            fresh_days=int(avail["fresh_days"]),
            recency_decay_window=int(avail["recency_decay_window"]),
            recency_floor=float(avail["recency_floor"]),
            not_open_factor=float(avail["not_open_factor"]),
            interview_floor=float(avail["interview_floor"]),
            offer_floor=float(avail["offer_floor"]),
            market_inactive_factor=float(avail["market_inactive_factor"]),
            avail_min=float(avail["avail_min"]),
        ),
        logistics=LogisticsConfig(
            loc_preferred_bonus=float(log["loc_preferred_bonus"]),
            loc_acceptable_bonus=float(log["loc_acceptable_bonus"]),
            loc_outside_penalty=float(log["loc_outside_penalty"]),
            workmode_fit_bonus=float(log["workmode_fit_bonus"]),
            workmode_remote_penalty=float(log["workmode_remote_penalty"]),
            notice_penalty_scale=float(log["notice_penalty_scale"]),
            notice_penalty_cap=float(log["notice_penalty_cap"]),
        ),
        enable_popularity_tiebreak=bool(s5.get("enable_popularity_tiebreak", False)),
        popularity_tiebreak_cap=float(s5.get("popularity_tiebreak_cap", 0.01)),
    )
