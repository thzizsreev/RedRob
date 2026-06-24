#!/usr/bin/env python3
"""
Stage 0 — cross-encoder ONNX export (offline, one-time).

Prerequisite for Stage 4. Independent of candidate pool — run anytime before Stage 4.

    pip install -r models/requirements.txt
    python tracks/instructor/stage0/run_cross_encoder.py

Outputs (under OUTPUT_DIR = models/cross_encoder/):
  model.onnx
  tokenizer/
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage0.cross_encoder_export import (
    load_model_id_from_config,
    run_cross_encoder_export,
)
from tracks.shared.paths import CROSS_ENCODER_DIR, ROOT_DIR

# --- edit before run ---
OUTPUT_DIR = CROSS_ENCODER_DIR
CONFIG_PATH = ROOT_DIR / "config.yaml"
SKIP_IF_EXISTS = True


def main() -> None:
    model_id = load_model_id_from_config(CONFIG_PATH)
    start_time = perf_counter()
    run_cross_encoder_export(
        model_id=model_id,
        output_dir=OUTPUT_DIR,
        skip_if_exists=SKIP_IF_EXISTS,
    )
    elapsed = perf_counter() - start_time
    print(f"Cross-encoder export completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
