#!/usr/bin/env python3
"""Stage 3 runner — retrieval + fusion using precomputed artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_stage_3.shared.config_runner import DEFAULT_CONFIG, load_runner_config
from test_stage_3.shared.io_precompute import load_manifest
from test_stage_3.shared.retrieve import print_stage3_summary, run

RUNNER_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 runner (loads precomputed query vectors, no ONNX)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runner_config(args.config)

    if not config.precomputed_manifest.exists():
        print(
            f"Missing {config.precomputed_manifest}. "
            "Run: python test_stage_3/precompute/run.py"
        )
        sys.exit(1)

    load_manifest(config.precomputed_manifest)
    start = perf_counter()
    result = run(config_path=args.config)
    print_stage3_summary(result)
    total = perf_counter() - start
    print(f"\nStage 3 runner completed in {total:.2f} seconds")


if __name__ == "__main__":
    main()
