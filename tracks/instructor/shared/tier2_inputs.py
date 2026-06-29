"""Tier 2 penalty/bonus column derivation for Stage 4 enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml


@dataclass(frozen=True)
class TitleChasingConfig:
    coef: float
    cap: float


@dataclass(frozen=True)
class AmbiguityConfig:
    title_weight: float
    near_band_weight: float


@dataclass(frozen=True)
class ClosedSourceConfig:
    value: float
    validation_threshold: float


@dataclass(frozen=True)
class OptionalBonusConfig:
    per_category: float
    cap: float
    categories: dict[str, tuple[str, ...]]
    oss_validation_threshold: float


@dataclass(frozen=True)
class Tier2InputsConfig:
    sweet_spot_bonus: float
    title_chasing: TitleChasingConfig
    ambiguity: AmbiguityConfig
    closed_source: ClosedSourceConfig
    optional_bonus: OptionalBonusConfig


def load_tier2_config(config_path: Path) -> Tier2InputsConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "stage5" not in raw:
        raise ValueError(f"Missing 'stage5' namespace in {config_path}")

    t2 = raw["stage5"]["tier2"]
    tc = t2["title_chasing"]
    amb = t2["ambiguity"]
    cs = t2["closed_source"]
    ob = t2["optional_bonus"]

    categories: dict[str, tuple[str, ...]] = {}
    for name, keywords in ob["categories"].items():
        categories[name] = tuple(str(k).lower() for k in keywords)

    return Tier2InputsConfig(
        sweet_spot_bonus=float(t2["sweet_spot_bonus"]),
        title_chasing=TitleChasingConfig(
            coef=float(tc["coef"]),
            cap=float(tc["cap"]),
        ),
        ambiguity=AmbiguityConfig(
            title_weight=float(amb["title_weight"]),
            near_band_weight=float(amb["near_band_weight"]),
        ),
        closed_source=ClosedSourceConfig(
            value=float(cs["value"]),
            validation_threshold=float(cs["validation_threshold"]),
        ),
        optional_bonus=OptionalBonusConfig(
            per_category=float(ob["per_category"]),
            cap=float(ob["cap"]),
            categories=categories,
            oss_validation_threshold=float(
                ob.get("oss_validation_threshold", cs["validation_threshold"])
            ),
        ),
    )


def _skill_names(skills: list[dict] | None) -> list[str]:
    if not skills:
        return []
    return [str(s.get("name", "")).lower() for s in skills if s.get("name")]


def compute_optional_bonus(
    skills: list[dict] | None,
    has_github: bool,
    external_validation_score: float | None,
    config: OptionalBonusConfig,
) -> float:
    names = _skill_names(skills)
    if not names:
        return 0.0

    ext_val = float(external_validation_score or 0.0)
    matched = 0
    for category, keywords in config.categories.items():
        if category == "oss":
            if not has_github or ext_val < config.oss_validation_threshold:
                continue
        if any(kw in name for name in names for kw in keywords):
            matched += 1

    return min(config.cap, matched * config.per_category)


def enrich_tier2_columns(
    df: pl.DataFrame,
    skills_by_id: dict[str, list[dict]],
    config: Tier2InputsConfig,
) -> pl.DataFrame:
    tc = config.title_chasing
    amb = config.ambiguity
    cs = config.closed_source

    work = df.with_columns(
        [
            pl.min_horizontal(
                pl.col("short_hop_count").cast(pl.Float64).fill_null(0) * pl.lit(tc.coef),
                pl.lit(tc.cap),
            ).alias("title_chasing_penalty"),
            (
                pl.col("title_ambiguous").cast(pl.Float64).fill_null(0) * pl.lit(amb.title_weight)
                + (pl.col("exp_band") == "near_band").cast(pl.Float64) * pl.lit(amb.near_band_weight)
            ).alias("ambiguity_penalty"),
            pl.when(
                (pl.col("external_validation_score").fill_null(0.0) < cs.validation_threshold)
                & (~pl.col("has_github").fill_null(False))
            )
            .then(pl.lit(cs.value))
            .otherwise(pl.lit(0.0))
            .alias("closed_source_penalty"),
        ]
    )

    bonuses: list[float] = []
    for row in work.iter_rows(named=True):
        cid = str(row["candidate_id"])
        skills = skills_by_id.get(cid, [])
        has_github = bool(row.get("has_github"))
        ext_val = row.get("external_validation_score")
        bonuses.append(
            compute_optional_bonus(skills, has_github, ext_val, config.optional_bonus)
        )

    return work.with_columns(pl.Series("optional_bonus", bonuses))
