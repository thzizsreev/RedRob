#!/usr/bin/env python3
"""Build docker/config.yaml from root config with sandbox overrides."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from docker.paths import POOL1K_JSONL, POOL1K_STAGE0, ROOT_DIR, SANDBOX_CONFIG, WORK_RUNTIME

POOL_SIZE = 1000
SUBMISSION_TOP_N = 100


def make_sandbox_config(
    *,
    n_pool: int = POOL_SIZE,
    full_pool: bool = True,
) -> dict:
    with open(ROOT_DIR / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    stage3_max_k = min(600, n_pool)
    stage4_keep = min(300, n_pool)

    cfg.setdefault("stage2", {})
    cfg["stage2"]["expected_input_count"] = n_pool
    cfg["stage2"]["expected_survivor_min"] = 1
    cfg["stage2"]["expected_survivor_max"] = n_pool

    cfg.setdefault("stage3", {})
    cfg["stage3"]["min_k"] = 1
    cfg["stage3"]["max_k"] = stage3_max_k
    cfg["stage3"]["expected_survivor_min"] = 1
    cfg["stage3"]["expected_survivor_max"] = n_pool

    cfg.setdefault("stage4", {})
    cfg["stage4"]["keep_n"] = stage4_keep
    cfg["stage4"]["expected_input_min"] = 1
    cfg["stage4"]["expected_input_max"] = stage3_max_k

    cfg.setdefault("stage5", {})
    cfg["stage5"]["top_n"] = SUBMISSION_TOP_N
    cfg["stage5"]["output_dir"] = "docker/work/runtime/stage5"

    if full_pool:
        stage0_rel = POOL1K_STAGE0.relative_to(ROOT_DIR).as_posix()
        jsonl_rel = POOL1K_JSONL.relative_to(ROOT_DIR).as_posix()
        features = f"{stage0_rel}/candidate_features.parquet"
        candidates = jsonl_rel
    else:
        features = "docker/work/runtime/stage0/candidate_features.parquet"
        candidates = "docker/work/data/active.jsonl"

    cfg["stage4"]["candidate_features_path"] = features
    cfg["stage4"]["candidates_jsonl_path"] = candidates
    cfg["stage5"]["candidate_features_path"] = features
    cfg["stage5"]["candidates_jsonl_path"] = candidates

    return cfg


def write_config(
    path: Path,
    *,
    n_pool: int = POOL_SIZE,
    full_pool: bool = True,
) -> Path:
    cfg = make_sandbox_config(n_pool=n_pool, full_pool=full_pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def main() -> None:
    write_config(SANDBOX_CONFIG, full_pool=True)
    write_config(WORK_RUNTIME.parent / "config.runtime.yaml", full_pool=True)
    print(f"Wrote {SANDBOX_CONFIG}")


if __name__ == "__main__":
    main()
