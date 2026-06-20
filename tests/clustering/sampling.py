"""Stage 1 — stratified sampling with optional landmark force-include."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.clustering.io import ClusteringInputs


@dataclass(frozen=True)
class SampledData:
    indices: np.ndarray
    candidate_ids: list[str]
    records: list[dict]
    vectors: np.ndarray


def _years_bucket(years: float) -> str:
    if years < 3:
        return "0-3"
    if years < 6:
        return "3-6"
    if years < 10:
        return "6-10"
    return "10+"


def _skill_count(record: dict) -> int:
    skills = record.get("skills", {})
    if isinstance(skills, list):
        return len(skills)
    if isinstance(skills, dict):
        return sum(len(skills.get(key, [])) for key in ("technical", "soft", "tools"))
    return 0


def _skill_bucket(count: int, tertiles: tuple[int, int]) -> str:
    low, high = tertiles
    if count <= low:
        return "low"
    if count <= high:
        return "medium"
    return "high"


def _stratum_key(record: dict, tertiles: tuple[int, int]) -> str:
    years = float(record.get("profile", {}).get("years_of_experience", 0))
    return f"{_years_bucket(years)}|{_skill_bucket(_skill_count(record), tertiles)}"


def stratified_sample(
    inputs: ClusteringInputs,
    sample_size: int,
    random_seed: int,
    landmark_ids: list[str],
) -> SampledData:
    n = len(inputs.candidate_ids)
    target = min(sample_size, n)

    skill_counts = [_skill_count(record) for record in inputs.records]
    tertiles = (
        int(np.percentile(skill_counts, 33)),
        int(np.percentile(skill_counts, 66)),
    )

    rng = np.random.default_rng(random_seed)
    strata: dict[str, list[int]] = {}
    for idx, record in enumerate(inputs.records):
        key = _stratum_key(record, tertiles)
        strata.setdefault(key, []).append(idx)

    selected: set[int] = set()
    landmark_indices = [
        inputs.candidate_ids.index(landmark_id)
        for landmark_id in landmark_ids
        if landmark_id in inputs.candidate_ids
    ]
    selected.update(landmark_indices)

    remaining = max(target - len(selected), 0)
    if remaining > 0:
        stratum_keys = list(strata.keys())
        rng.shuffle(stratum_keys)
        per_stratum = max(1, remaining // len(stratum_keys))

        for key in stratum_keys:
            pool = [idx for idx in strata[key] if idx not in selected]
            if not pool:
                continue
            take = min(per_stratum, len(pool), remaining)
            chosen = rng.choice(pool, size=take, replace=False)
            selected.update(int(i) for i in chosen)
            remaining -= take
            if remaining <= 0:
                break

        if remaining > 0:
            pool = [idx for idx in range(n) if idx not in selected]
            if pool:
                extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                selected.update(int(i) for i in extra)

    indices = np.array(sorted(selected), dtype=np.int64)
    return SampledData(
        indices=indices,
        candidate_ids=[inputs.candidate_ids[i] for i in indices],
        records=[inputs.records[i] for i in indices],
        vectors=inputs.vectors[indices],
    )
