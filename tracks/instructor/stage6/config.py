"""Load Stage 6 configuration from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tracks.shared.paths import (
    CANDIDATES_JSONL_PATH,
    PARAPHRASER_DIR,
    PRECOMPUTED_DIR,
    REASONING_LOOKUP_PATH,
    REASONING_RAW_PATH,
    ROOT_DIR,
    RUNTIME_STAGE5_DIR,
    RUNTIME_STAGE6_DIR,
)


@dataclass(frozen=True)
class Stage6Config:
    team_id: str
    top_n: int
    stage5_top100_path: Path
    candidates_jsonl_path: Path
    reasoning_raw_path: Path
    reasoning_lookup_path: Path
    use_reasoning_lookup: str
    candidate_features_path: Path
    paraphraser_dir: Path
    output_dir: Path
    ort_intra_op_threads: int
    estimated_session_mb: int
    memory_reserve_ratio: float
    max_workers: int | None
    model_id: str
    input_prefix: str
    max_length: int
    top_p: float
    repetition_penalty: float


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def load_stage6_config(config_path: Path) -> Stage6Config:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "stage6" not in raw:
        raise ValueError(f"Missing 'stage6' namespace in {config_path}")

    s6 = raw["stage6"]
    max_workers = s6.get("max_workers")
    return Stage6Config(
        team_id=str(s6["team_id"]),
        top_n=int(s6.get("top_n", 100)),
        stage5_top100_path=_resolve_path(
            str(s6.get("stage5_top100_path", RUNTIME_STAGE5_DIR / "stage5_scored_top100.parquet"))
        ),
        candidates_jsonl_path=_resolve_path(
            str(s6.get("candidates_jsonl_path", CANDIDATES_JSONL_PATH))
        ),
        reasoning_raw_path=_resolve_path(
            str(s6.get("reasoning_raw_path", REASONING_RAW_PATH))
        ),
        reasoning_lookup_path=_resolve_path(
            str(s6.get("reasoning_lookup_path", REASONING_LOOKUP_PATH))
        ),
        use_reasoning_lookup=str(s6.get("use_reasoning_lookup", "auto")),
        candidate_features_path=_resolve_path(
            str(s6.get("candidate_features_path", PRECOMPUTED_DIR / "candidate_features.parquet"))
        ),
        paraphraser_dir=_resolve_path(str(s6.get("paraphraser_dir", PARAPHRASER_DIR))),
        output_dir=_resolve_path(str(s6.get("output_dir", RUNTIME_STAGE6_DIR))),
        ort_intra_op_threads=int(s6.get("ort_intra_op_threads", 1)),
        estimated_session_mb=int(s6.get("estimated_session_mb", 700)),
        memory_reserve_ratio=float(s6.get("memory_reserve_ratio", 0.25)),
        max_workers=int(max_workers) if max_workers is not None else None,
        model_id=str(s6.get("model_id", "humarin/chatgpt_paraphraser_on_T5_base")),
        input_prefix=str(s6.get("input_prefix", "paraphrase: ")),
        max_length=int(s6.get("max_length", 128)),
        top_p=float(s6.get("top_p", 0.92)),
        repetition_penalty=float(s6.get("repetition_penalty", 1.3)),
    )
