"""Load Stage 4 configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tracks.shared.paths import CANDIDATES_JSONL_PATH, CROSS_ENCODER_DIR, ROOT_DIR


@dataclass(frozen=True)
class Stage4Config:
    model_id: str
    onnx_model_path: Path
    tokenizer_path: Path
    jd_text: str
    max_jd_tokens: int
    max_candidate_tokens: int
    max_pair_tokens: int
    batch_size: int
    num_threads: int
    keep_n: int
    rank_delta_threshold: int
    expected_input_min: int
    expected_input_max: int
    candidate_features_path: Path
    candidates_jsonl_path: Path
    empty_score: float = -1e9


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def load_stage4_config(config_path: Path) -> Stage4Config:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "stage4" not in raw:
        raise ValueError(f"Missing 'stage4' namespace in {config_path}")

    s4 = raw["stage4"]
    onnx_path = _resolve_path(str(s4.get("onnx_model_path", CROSS_ENCODER_DIR / "model.onnx")))
    tokenizer_path = _resolve_path(
        str(s4.get("tokenizer_path", CROSS_ENCODER_DIR / "tokenizer"))
    )

    return Stage4Config(
        model_id=str(s4["model_id"]),
        onnx_model_path=onnx_path,
        tokenizer_path=tokenizer_path,
        jd_text=str(s4["jd_text"]).strip(),
        max_jd_tokens=int(s4.get("max_jd_tokens", 256)),
        max_candidate_tokens=int(s4.get("max_candidate_tokens", 384)),
        max_pair_tokens=int(s4.get("max_pair_tokens", 512)),
        batch_size=int(s4.get("batch_size", 16)),
        num_threads=int(s4.get("num_threads", 4)),
        keep_n=int(s4.get("keep_n", 300)),
        rank_delta_threshold=int(s4.get("rank_delta_threshold", 50)),
        expected_input_min=int(s4.get("expected_input_min", 300)),
        expected_input_max=int(s4.get("expected_input_max", 600)),
        candidate_features_path=_resolve_path(
            str(s4.get("candidate_features_path", "artifacts/precomputed/candidate_features.parquet"))
        ),
        candidates_jsonl_path=_resolve_path(
            str(s4.get("candidates_jsonl_path", CANDIDATES_JSONL_PATH))
        ),
    )
