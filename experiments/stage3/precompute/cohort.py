"""Build cohort fixtures: stage2 slice + skill_weighted_score stub."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from tracks.instructor.core.io import load_index_and_id_map
from experiments.stage3.shared.config_precompute import PrecomputeConfig
from experiments.stage3.shared.io_stage0 import build_survivor_row_indices, load_retrieval_assets

PROFICIENCY_SCORES = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}
TIER_REWARDS = {
    "retrieval": 1.0,
    "vector_db": 0.85,
    "eval": 0.75,
    "python": 0.5,
}


def _skill_tier(skill_name: str, keywords: dict[str, list[str]]) -> str | None:
    name = skill_name.lower()
    for tier, kws in keywords.items():
        if any(kw in name for kw in kws):
            return tier
    return None


def _depth_score(proficiency: str | None, duration_months: int | None) -> float:
    prof = PROFICIENCY_SCORES.get(str(proficiency or "").lower(), 0.5)
    years = max(float(duration_months or 0) / 12.0, 0.0)
    duration_part = min(math.log(1.0 + years) / math.log(11.0), 1.0)
    return 0.6 * duration_part + 0.4 * prof


def _stub_skill_score(skills: list[dict], keywords: dict[str, list[str]]) -> float:
    if not skills:
        return 0.0

    scored: list[tuple[float, float]] = []
    for skill in skills:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        tier = _skill_tier(name, keywords)
        if tier is None:
            continue
        relevance = TIER_REWARDS.get(tier, 0.25)
        depth = _depth_score(skill.get("proficiency"), skill.get("duration_months"))
        scored.append((depth, relevance * depth))

    if not scored:
        return 0.0

    scored.sort(key=lambda x: -x[0])
    return sum(d for _, d in scored[:15])


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _load_candidates_for_ids(
    candidates_path: Path,
    wanted: set[str],
) -> dict[str, list[dict]]:
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidates file: {candidates_path}")

    found: dict[str, list[dict]] = {}
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = str(record.get("candidate_id", ""))
            if cid not in wanted or cid in found:
                continue
            found[cid] = record.get("skills") or []
            if len(found) == len(wanted):
                break

    missing = wanted - set(found.keys())
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} sample candidate(s) missing from jsonl. Examples: {examples}"
        )
    return found


def build_cohort(config: PrecomputeConfig, cohort_dir: Path) -> pl.DataFrame:
    cohort_dir.mkdir(parents=True, exist_ok=True)

    if not config.stage2_source_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 source: {config.stage2_source_path}")

    stage2_full = pl.read_parquet(config.stage2_source_path)
    sample = stage2_full.sample(
        n=min(config.sample_size, stage2_full.height),
        seed=config.random_seed,
    )
    stage2_path = cohort_dir / "stage2_gated.parquet"
    sample.write_parquet(stage2_path)
    print(f"Wrote {stage2_path} ({sample.height:,} rows)")

    _, id_map = load_index_and_id_map(config.stage0_dir)
    id_set = set(id_map.values())
    sample_ids = set(sample["candidate_id"].cast(pl.Utf8).to_list())
    unknown = sample_ids - id_set
    if unknown:
        examples = sorted(unknown)[:5]
        raise ValueError(
            f"{len(unknown)} sample IDs not in id_map. Examples: {examples}"
        )

    skills_by_id = _load_candidates_for_ids(config.candidates_jsonl_path, sample_ids)
    raw_scores: list[tuple[str, float]] = []
    for cid in sorted(sample_ids):
        raw = _stub_skill_score(skills_by_id[cid], config.must_have_keywords)
        raw_scores.append((cid, raw))

    normalized = _min_max_normalize([s for _, s in raw_scores])
    features_rows = [
        {"candidate_id": cid, "skill_weighted_score": score}
        for (cid, _), score in zip(raw_scores, normalized)
    ]
    features_df = pl.DataFrame(features_rows)
    features_path = cohort_dir / "candidate_features.parquet"
    features_df.write_parquet(features_path)
    print(f"Wrote {features_path} ({features_df.height:,} rows)")

    return sample


def build_survivor_indices(
    config: PrecomputeConfig,
    stage2_df: pl.DataFrame,
    cohort_dir: Path,
) -> np.ndarray:
    assets = load_retrieval_assets(config.stage0_dir)
    indices = build_survivor_row_indices(stage2_df, assets.id_to_row)
    out_path = cohort_dir / "survivor_row_indices.npy"
    np.save(out_path, indices)
    print(f"Wrote {out_path} ({indices.size:,} indices)")
    return indices


def write_stage0_pointer(config: PrecomputeConfig, artifacts_dir: Path) -> Path:
    pointer_path = artifacts_dir / "stage0_pointer.json"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    from tracks.shared.paths import ROOT_DIR

    try:
        rel = config.stage0_dir.relative_to(ROOT_DIR)
        stage0_dir = str(rel).replace("\\", "/")
    except ValueError:
        stage0_dir = str(config.stage0_dir)

    payload = {"stage0_dir": stage0_dir}
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {pointer_path}")
    return pointer_path
