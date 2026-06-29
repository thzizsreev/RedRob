"""Build Q1/Q2 vectors from one facet-centroid formula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from tracks.instructor.stage3.config import FacetSpec, SubspaceWeights as Stage3SubspaceWeights
from tracks.instructor.stage3.query_encode import build_centroid_query


@dataclass(frozen=True)
class SubspaceWeights:
    retrieval: float
    infra: float
    eval: float


@dataclass(frozen=True)
class VectorConfig:
    q1_subspace: SubspaceWeights
    q2_subspace: SubspaceWeights
    q1_facets: tuple[FacetSpec, ...]
    q2_facets: tuple[FacetSpec, ...]


def parse_subspace_weights(raw: dict[str, float]) -> SubspaceWeights:
    return SubspaceWeights(
        retrieval=float(raw["retrieval"]),
        infra=float(raw["infra"]),
        eval=float(raw["eval"]),
    )


def _to_stage3_weights(weights: SubspaceWeights) -> Stage3SubspaceWeights:
    return Stage3SubspaceWeights(
        retrieval=weights.retrieval,
        infra=weights.infra,
        eval=weights.eval,
    )


def build_q1_vector(model: Any, config: VectorConfig):
    return build_centroid_query(
        model,
        config.q1_facets,
        _to_stage3_weights(config.q1_subspace),
    )


def build_q2_vector(model: Any, config: VectorConfig):
    return build_centroid_query(
        model,
        config.q2_facets,
        _to_stage3_weights(config.q2_subspace),
    )


def _facet_tuple(
    texts: dict[str, str],
    weights: dict[str, float],
) -> tuple[FacetSpec, ...]:
    return tuple(
        FacetSpec(id=fid, text=texts[fid], weight=float(weights[fid]))
        for fid in weights
    )


def load_vector_config(path: str | Any) -> VectorConfig:
    with open(path, encoding="utf-8") as f:
        root = yaml.safe_load(f)

    sub = root["subspace"]
    facets = root["facets"]
    weights = root["weights"]

    q1_texts = {k: str(v).strip() for k, v in facets["q1"].items()}
    q2_texts = {k: str(v).strip() for k, v in facets["q2"].items()}

    return VectorConfig(
        q1_subspace=parse_subspace_weights(sub["q1"]),
        q2_subspace=parse_subspace_weights(sub["q2"]),
        q1_facets=_facet_tuple(q1_texts, weights["q1"]),
        q2_facets=_facet_tuple(q2_texts, weights["q2"]),
    )
