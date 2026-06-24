"""Stage 0 — INSTRUCTOR-large ONNX block-weighted vector precomputation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tracks.instructor.index import build_vector_index, build_vector_index_from_records
from tracks.instructor.stage0.bm25_precompute import build_bm25_index


def load_candidates_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def _load_q4_tokens(config_path: Path | None) -> list[str] | None:
    if config_path is None or not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    stage3 = raw.get("stage3", {})
    tokens = stage3.get("q4_tokens")
    if not tokens:
        return None
    return [str(t) for t in tokens]


def run_precompute(
    candidates_path: Path,
    model,
    output_dir: Path,
    *,
    config_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to {output_dir}")

    if candidates_path.suffix == ".json":
        candidates = load_candidates_json(candidates_path)
        print(f"Processing {len(candidates)} candidates from {candidates_path}...")
        _, _, _, records = build_vector_index_from_records(candidates, model, output_dir)
    else:
        print(f"Processing candidates from {candidates_path}...")
        _, _, _, records = build_vector_index(candidates_path, model, output_dir)

    q4_tokens = _load_q4_tokens(config_path)
    print("Building BM25 jargon index...")
    build_bm25_index(records, output_dir, q4_tokens=q4_tokens)

    print("Precompute complete.")
