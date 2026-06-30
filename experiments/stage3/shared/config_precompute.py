"""Load precompute/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tracks.shared.paths import ROOT_DIR

PRECOMPUTE_DIR = ROOT_DIR / "experiments" / "stage3" / "precompute"
DEFAULT_CONFIG = PRECOMPUTE_DIR / "config.yaml"


@dataclass(frozen=True)
class SubspaceWeights:
    retrieval: float
    infra: float
    eval: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.retrieval, self.infra, self.eval)


@dataclass(frozen=True)
class PrecomputeConfig:
    stage0_dir: Path
    stage2_source_path: Path
    candidates_jsonl_path: Path
    artifacts_dir: Path
    sample_size: int
    random_seed: int
    q1_text: str
    q2_text: str
    q3_text: str
    subspace_weights_q1: SubspaceWeights
    subspace_weights_q2: SubspaceWeights
    subspace_weights_q3: SubspaceWeights
    must_have_keywords: dict[str, list[str]]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _parse_weights(raw: dict) -> SubspaceWeights:
    return SubspaceWeights(
        retrieval=float(raw["retrieval"]),
        infra=float(raw["infra"]),
        eval=float(raw["eval"]),
    )


def load_precompute_config(config_path: Path | None = None) -> PrecomputeConfig:
    path = config_path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "precompute" not in raw:
        raise ValueError(f"Missing 'precompute' namespace in {path}")

    pc = raw["precompute"]
    return PrecomputeConfig(
        stage0_dir=_resolve_path(str(pc["stage0_dir"])),
        stage2_source_path=_resolve_path(str(pc["stage2_source_path"])),
        candidates_jsonl_path=_resolve_path(str(pc["candidates_jsonl_path"])),
        artifacts_dir=_resolve_path(str(pc["artifacts_dir"])),
        sample_size=int(pc["sample_size"]),
        random_seed=int(pc["random_seed"]),
        q1_text=str(pc["q1_text"]).strip(),
        q2_text=str(pc["q2_text"]).strip(),
        q3_text=str(pc["q3_text"]).strip(),
        subspace_weights_q1=_parse_weights(pc["subspace_weights_q1"]),
        subspace_weights_q2=_parse_weights(pc["subspace_weights_q2"]),
        subspace_weights_q3=_parse_weights(pc["subspace_weights_q3"]),
        must_have_keywords={
            str(k): [str(v) for v in vals]
            for k, vals in (pc.get("must_have_keywords") or {}).items()
        },
    )
