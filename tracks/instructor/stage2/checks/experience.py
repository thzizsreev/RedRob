"""Check A — experience band gate."""

from __future__ import annotations

from dataclasses import dataclass

from tracks.instructor.stage2.config import Stage2Config


@dataclass(frozen=True)
class ExperienceResult:
    total_years_exp: float
    exp_band: str | None
    in_sweet_spot: bool
    remove: bool
    reason: str | None


def evaluate_experience(record: dict, config: Stage2Config) -> ExperienceResult:
    profile = record.get("profile") or {}
    years = profile.get("years_of_experience")
    if years is None:
        return ExperienceResult(
            total_years_exp=0.0,
            exp_band=None,
            in_sweet_spot=False,
            remove=True,
            reason="exp_out_of_band",
        )

    total = float(years)
    hard_lo = config.hard_min - config.soft_tolerance
    hard_hi = config.hard_max + config.soft_tolerance

    in_sweet_spot = config.sweet_low <= total <= config.sweet_high

    if total < hard_lo or total > hard_hi:
        return ExperienceResult(
            total_years_exp=total,
            exp_band=None,
            in_sweet_spot=in_sweet_spot,
            remove=True,
            reason="exp_out_of_band",
        )

    if config.hard_min <= total <= config.hard_max:
        exp_band = "in_band"
    else:
        exp_band = "near_band"

    return ExperienceResult(
        total_years_exp=total,
        exp_band=exp_band,
        in_sweet_spot=in_sweet_spot,
        remove=False,
        reason=None,
    )
