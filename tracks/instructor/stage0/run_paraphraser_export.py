#!/usr/bin/env python3
"""
Stage 0 — paraphraser ONNX encoder export (offline, one-time).

    pip install -r models/requirements.txt
    python tracks/instructor/stage0/run_paraphraser_export.py

Outputs (under models/paraphraser/):
  encoder.onnx
  tokenizer/
  pytorch/          # full weights for CPU decoder fallback
  manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

_EXPORT_PATH = _ROOT / "tracks" / "instructor" / "stage0" / "paraphraser_export.py"
_spec = importlib.util.spec_from_file_location("paraphraser_export", _EXPORT_PATH)
assert _spec and _spec.loader
_pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pe)

load_model_id_from_config = _pe.load_model_id_from_config
run_paraphraser_export = _pe.run_paraphraser_export
from tracks.shared.paths import PARAPHRASER_DIR, ROOT_DIR

OUTPUT_DIR = PARAPHRASER_DIR
CONFIG_PATH = ROOT_DIR / "config.yaml"
SKIP_IF_EXISTS = True


def main() -> None:
    model_id = load_model_id_from_config(CONFIG_PATH)
    start = perf_counter()
    run_paraphraser_export(
        model_id=model_id,
        output_dir=OUTPUT_DIR,
        skip_if_exists=SKIP_IF_EXISTS,
        config_path=CONFIG_PATH,
    )
    print(f"Paraphraser export completed in {perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()
