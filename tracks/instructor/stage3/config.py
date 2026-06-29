"""Load Stage 3 configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SubspaceWeights:
    retrieval: float
    infra: float
    eval: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.retrieval, self.infra, self.eval)


@dataclass(frozen=True)
class FacetSpec:
    id: str
    text: str
    weight: float


@dataclass(frozen=True)
class Stage3Config:
    q1_facets: tuple[FacetSpec, ...]
    q1_text: str | None
    q2_text: str
    q3_text: str
    subspace_weights_q1: SubspaceWeights
    subspace_weights_q2: SubspaceWeights
    subspace_weights_q3: SubspaceWeights
    per_query_k_dense: int
    per_query_k_skill: int
    miss_penalty_dense: int
    miss_penalty_skill: int
    rrf_k: int
    alpha_neg: float
    beta_cluster: float
    z_threshold: float
    min_k: int
    max_k: int
    use_prf: bool
    expected_survivor_min: int
    expected_survivor_max: int


def _parse_weights(raw: dict) -> SubspaceWeights:
    return SubspaceWeights(
        retrieval=float(raw["retrieval"]),
        infra=float(raw["infra"]),
        eval=float(raw["eval"]),
    )


def _parse_q1_facets(s3: dict) -> tuple[tuple[FacetSpec, ...], str | None]:
    if "q1_facets" in s3 and "q1_facet_weights" in s3:
        texts = {k: str(v).strip() for k, v in s3["q1_facets"].items()}
        weights = s3["q1_facet_weights"]
        facets = tuple(
            FacetSpec(id=fid, text=texts[fid], weight=float(weights[fid]))
            for fid in weights
        )
        weight_sum = sum(f.weight for f in facets)
        if abs(weight_sum - 1.0) > 1e-4:
            raise ValueError(f"q1_facet_weights must sum to 1.0, got {weight_sum}")
        return facets, None
    if "q1_text" in s3:
        return (), str(s3["q1_text"]).strip()
    raise ValueError(
        "stage3 requires q1_facets + q1_facet_weights, or legacy q1_text"
    )


def load_stage3_config(config_path: Path) -> Stage3Config:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "stage3" not in raw:
        raise ValueError(f"Missing 'stage3' namespace in {config_path}")

    s3 = raw["stage3"]
    q1_facets, q1_text = _parse_q1_facets(s3)
    per_query_k_dense = int(s3["per_query_k_dense"])
    per_query_k_skill = int(s3["per_query_k_skill"])
    return Stage3Config(
        q1_facets=q1_facets,
        q1_text=q1_text,
        q2_text=str(s3["q2_text"]).strip(),
        q3_text=str(s3["q3_text"]).strip(),
        subspace_weights_q1=_parse_weights(s3["subspace_weights_q1"]),
        subspace_weights_q2=_parse_weights(s3["subspace_weights_q2"]),
        subspace_weights_q3=_parse_weights(s3["subspace_weights_q3"]),
        per_query_k_dense=per_query_k_dense,
        per_query_k_skill=per_query_k_skill,
        miss_penalty_dense=int(s3.get("miss_penalty_dense", per_query_k_dense + 1)),
        miss_penalty_skill=int(s3.get("miss_penalty_skill", per_query_k_skill + 1)),
        rrf_k=int(s3["rrf_k"]),
        alpha_neg=float(s3["alpha_neg"]),
        beta_cluster=float(s3["beta_cluster"]),
        z_threshold=float(s3["z_threshold"]),
        min_k=int(s3["min_k"]),
        max_k=int(s3["max_k"]),
        use_prf=bool(s3.get("use_prf", False)),
        expected_survivor_min=int(s3.get("expected_survivor_min", 2000)),
        expected_survivor_max=int(s3.get("expected_survivor_max", 5000)),
    )
