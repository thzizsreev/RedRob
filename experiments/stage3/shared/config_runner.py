"""Load runner/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tracks.shared.paths import ROOT_DIR

RUNNER_DIR = ROOT_DIR / "experiments" / "stage3" / "runner"
DEFAULT_CONFIG = RUNNER_DIR / "config.yaml"


@dataclass(frozen=True)
class RunnerConfig:
    precomputed_manifest: Path
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
    output_dir: Path
    expected_survivor_min: int
    expected_survivor_max: int


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def load_runner_config(config_path: Path | None = None) -> RunnerConfig:
    path = config_path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "runner" not in raw:
        raise ValueError(f"Missing 'runner' namespace in {path}")

    r = raw["runner"]
    per_query_k_dense = int(r["per_query_k_dense"])
    per_query_k_skill = int(r["per_query_k_skill"])

    return RunnerConfig(
        precomputed_manifest=_resolve_path(str(r["precomputed_manifest"])),
        per_query_k_dense=per_query_k_dense,
        per_query_k_skill=per_query_k_skill,
        miss_penalty_dense=int(r.get("miss_penalty_dense", per_query_k_dense + 1)),
        miss_penalty_skill=int(r.get("miss_penalty_skill", per_query_k_skill + 1)),
        rrf_k=int(r["rrf_k"]),
        alpha_neg=float(r["alpha_neg"]),
        beta_cluster=float(r["beta_cluster"]),
        z_threshold=float(r["z_threshold"]),
        min_k=int(r["min_k"]),
        max_k=int(r["max_k"]),
        output_dir=_resolve_path(str(r["output_dir"])),
        expected_survivor_min=int(r.get("expected_survivor_min", 400)),
        expected_survivor_max=int(r.get("expected_survivor_max", 600)),
    )
