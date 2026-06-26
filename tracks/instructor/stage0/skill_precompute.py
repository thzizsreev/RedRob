"""Global skill_weighted_score precompute — writes candidate_features.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from tracks.instructor.core.config import CANDIDATE_FEATURES_FILENAME
from tracks.instructor.stage0.skill_config import Stage0SkillConfig, load_stage0_skill_config
from tracks.instructor.stage0.skill_depth import depth_score
from tracks.instructor.stage0.skill_idf import build_skill_idf
from tracks.instructor.stage0.tier_relevance import build_tier_lookup, relevance_score


def _raw_skill_score(
    skills: list[dict],
    idf_table: dict[str, float],
    tier_lookup: dict[str, float],
    top_k: int,
) -> float:
    if not skills:
        return 0.0

    scored: list[tuple[float, float]] = []
    for skill in skills:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        relevance = relevance_score(name, tier_lookup)
        if relevance <= 0.0:
            continue
        depth = depth_score(skill.get("proficiency"), skill.get("duration_months"))
        rarity = idf_table.get(name.lower(), 0.0)
        sort_key = depth * rarity
        contribution = relevance * depth * rarity
        scored.append((sort_key, contribution))

    if not scored:
        return 0.0

    scored.sort(key=lambda x: -x[0])
    return sum(c for _, c in scored[:top_k])


def _p95_clip_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    p95 = float(np.percentile(arr, 95))
    clipped = np.minimum(arr, p95)
    lo = float(clipped.min())
    hi = float(clipped.max())
    if hi == lo:
        return [0.5] * len(values)
    return ((clipped - lo) / (hi - lo)).tolist()


def run_skill_precompute(
    records: list[dict],
    output_dir: Path,
    config_path: Path,
) -> Path:
    """Compute skill_weighted_score for all candidates and write parquet."""
    skill_config = load_stage0_skill_config(config_path)
    idf_path = output_dir / "skill_idf.json"
    idf_table, _ = build_skill_idf(records, persist_path=idf_path)
    tier_lookup = build_tier_lookup(skill_config)

    raw_scores: list[tuple[str, float]] = []
    for record in records:
        cid = str(record.get("candidate_id", ""))
        skills = record.get("skills") or []
        raw = _raw_skill_score(
            skills,
            idf_table,
            tier_lookup,
            skill_config.top_k_skills,
        )
        raw_scores.append((cid, raw))

    normalized = _p95_clip_minmax([s for _, s in raw_scores])
    rows = [
        {"candidate_id": cid, "skill_weighted_score": score}
        for (cid, _), score in zip(raw_scores, normalized)
    ]
    features_df = pl.DataFrame(rows)
    out_path = output_dir / CANDIDATE_FEATURES_FILENAME
    features_df.write_parquet(out_path)
    print(f"Wrote {out_path} ({features_df.height:,} rows)")
    return out_path
