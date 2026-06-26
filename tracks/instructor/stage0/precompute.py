"""Stage 0 — INSTRUCTOR-large ONNX block-weighted vector precomputation."""

from __future__ import annotations

import json
from pathlib import Path

from tracks.instructor.core.index import build_vector_index, build_vector_index_from_records


def load_candidates_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def run_precompute(
    candidates_path: Path,
    model,
    output_dir: Path,
) -> list[dict]:
    """Build FAISS index and candidate vectors. Returns records in row order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to {output_dir}")

    if candidates_path.suffix == ".json":
        candidates = load_candidates_json(candidates_path)
        print(f"Processing {len(candidates)} candidates from {candidates_path}...")
        _, _, _, records = build_vector_index_from_records(candidates, model, output_dir)
    else:
        print(f"Processing candidates from {candidates_path}...")
        _, _, _, records = build_vector_index(candidates_path, model, output_dir)

    print("Vector precompute complete.")
    return records
