"""Load Stage 5 v2 cascade configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from tracks.instructor.shared.tier2_inputs import Tier2InputsConfig, load_tier2_config
from tracks.shared.paths import CANDIDATES_JSONL_PATH, ROOT_DIR, RUNTIME_STAGE4_DIR, RUNTIME_STAGE5_DIR


@dataclass(frozen=True)
class BordaConfig:
    w_ce: float
    w_q1: float
    w_q2: float
    q_amplification_exponent: float


@dataclass(frozen=True)
class CascadeConfig:
    tier2_ratio: float
    tier3_ratio: float
    tier4_ratio: float


@dataclass(frozen=True)
class AvailabilityConfig:
    tier_a_interview_min: float
    tier_a_offer_min: float
    tier_a_recency_max_days: int
    tier_c_interview_max: float
    tier_c_recency_min_days: int


@dataclass(frozen=True)
class LogisticsConfig:
    location_units: dict[str, int]
    workmode_match_unit: int
    notice_short_unit: int
    notice_medium_unit: int
    notice_long_unit: int
    notice_short_max_days: int
    notice_long_min_days: int


@dataclass(frozen=True)
class Stage5Config:
    team_id: str
    top_n: int
    current_date: date
    stage4_input_path: Path
    candidates_jsonl_path: Path
    candidate_features_path: Path
    output_dir: Path
    borda: BordaConfig
    cascade: CascadeConfig
    tier2: Tier2InputsConfig
    availability: AvailabilityConfig
    logistics: LogisticsConfig


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _parse_current_date(value: str) -> date:
    if str(value).lower() == "auto":
        return date.today()
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def load_stage5_config(config_path: Path) -> Stage5Config:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "stage5" not in raw:
        raise ValueError(f"Missing 'stage5' namespace in {config_path}")

    s5 = raw["stage5"]
    borda_raw = s5.get("borda") or s5.get("borda_weights", {})
    borda = BordaConfig(
        w_ce=float(borda_raw["w_ce"]),
        w_q1=float(borda_raw["w_q1"]),
        w_q2=float(borda_raw["w_q2"]),
        q_amplification_exponent=float(borda_raw.get("q_amplification_exponent", 1.4)),
    )
    weight_sum = borda.w_ce + borda.w_q1 + borda.w_q2
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"stage5.borda weights must sum to 1.0, got {weight_sum}")

    cascade = CascadeConfig(
        tier2_ratio=float(s5["cascade"]["tier2_ratio"]),
        tier3_ratio=float(s5["cascade"]["tier3_ratio"]),
        tier4_ratio=float(s5["cascade"]["tier4_ratio"]),
    )

    avail = s5["availability"]
    log = s5["logistics"]
    location_units = {str(k): int(v) for k, v in log["location_units"].items()}

    return Stage5Config(
        team_id=str(s5["team_id"]),
        top_n=int(s5.get("top_n", 100)),
        current_date=_parse_current_date(str(s5["current_date"])),
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
        borda=borda,
        cascade=cascade,
        tier2=load_tier2_config(config_path),
        availability=AvailabilityConfig(
            tier_a_interview_min=float(avail["tier_a_interview_min"]),
            tier_a_offer_min=float(avail["tier_a_offer_min"]),
            tier_a_recency_max_days=int(avail["tier_a_recency_max_days"]),
            tier_c_interview_max=float(avail["tier_c_interview_max"]),
            tier_c_recency_min_days=int(avail["tier_c_recency_min_days"]),
        ),
        logistics=LogisticsConfig(
            location_units=location_units,
            workmode_match_unit=int(log["workmode_match_unit"]),
            notice_short_unit=int(log["notice_short_unit"]),
            notice_medium_unit=int(log["notice_medium_unit"]),
            notice_long_unit=int(log["notice_long_unit"]),
            notice_short_max_days=int(log["notice_short_max_days"]),
            notice_long_min_days=int(log["notice_long_min_days"]),
        ),
    )
