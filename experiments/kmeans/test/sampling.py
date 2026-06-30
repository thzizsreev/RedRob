"""Subsample candidate pool for K-means precompute."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from experiments.kmeans.io import KMeansInputs


@dataclass(frozen=True)
class Sample:
    candidate_ids: list[str]
    records: list[dict]
    vectors: np.ndarray


def subsample_inputs(
    inputs: KMeansInputs,
    sample_size: int,
    random_seed: int,
) -> Sample:
    n = len(inputs.candidate_ids)
    if sample_size >= n:
        return Sample(
            candidate_ids=inputs.candidate_ids,
            records=inputs.records,
            vectors=inputs.vectors,
        )

    rng = random.Random(random_seed)
    indices = rng.sample(range(n), k=sample_size)
    indices.sort()
    return Sample(
        candidate_ids=[inputs.candidate_ids[i] for i in indices],
        records=[inputs.records[i] for i in indices],
        vectors=inputs.vectors[indices],
    )
