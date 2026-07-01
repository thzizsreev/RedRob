"""CPU ONNX paraphraser — delegates to tracks.instructor.stage6."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.stage6.config import load_stage6_config
from tracks.instructor.stage6.paraphrase_onnx import (
    init_paraphraser,
    make_paraphrase_fn,
    paraphrase_once as _paraphrase_once,
)
from tracks.instructor.stage6.reasoning_builder import TEMPERATURES, pick_temperature, stable_seed
from tracks.shared.paths import PARAPHRASER_DIR, ROOT_DIR

MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"


def load_paraphraser() -> Callable[[str, float], str]:
    config_path = ROOT_DIR / "config.yaml"
    config = load_stage6_config(config_path)
    init_paraphraser(config)
    return make_paraphrase_fn()


__all__ = [
    "MODEL_NAME",
    "TEMPERATURES",
    "load_paraphraser",
    "pick_temperature",
    "stable_seed",
    "_paraphrase_once",
]
