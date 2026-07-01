#!/usr/bin/env python3
"""
Stage 0 — reasoning raw sentence precompute (pool-wide s1/s2).

    python tracks/instructor/stage0/run_reasoning_raw_precompute.py

Output: artifacts/precomputed/reasoning_raw.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

_PRECOMPUTE_PATH = _ROOT / "tracks" / "instructor" / "stage0" / "reasoning_raw_precompute.py"
_spec = importlib.util.spec_from_file_location("reasoning_raw_precompute", _PRECOMPUTE_PATH)
assert _spec and _spec.loader
_rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rp)

run_from_config = _rp.run_from_config
from tracks.shared.paths import ROOT_DIR

CONFIG_PATH = ROOT_DIR / "config.yaml"


def main() -> None:
    start = perf_counter()
    out = run_from_config(CONFIG_PATH)
    print(f"Reasoning raw precompute completed in {perf_counter() - start:.2f}s -> {out}")


if __name__ == "__main__":
    main()
