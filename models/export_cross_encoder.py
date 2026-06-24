#!/usr/bin/env python3
"""
Backward-compatible shim — prefer tracks/instructor/stage0/run_cross_encoder.py.

    pip install -r models/requirements.txt
    python models/export_cross_encoder.py

Outputs under models/cross_encoder/:
  model.onnx, tokenizer/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage0.cross_encoder_export import (
    DEFAULT_MODEL_ID,
    export_cross_encoder,
    print_manifest,
    run_cross_encoder_export,
    smoke_test,
)
from tracks.shared.paths import CROSS_ENCODER_DIR

OUTPUT_DIR = CROSS_ENCODER_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cross-encoder to ONNX for Stage 4.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export even when model.onnx and tokenizer/ already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cross_encoder_export(
        model_id=args.model_id,
        output_dir=args.output_dir,
        skip_if_exists=not args.overwrite,
    )


# Re-export for callers that imported symbols from this module
export_onnx = export_cross_encoder

if __name__ == "__main__":
    main()
