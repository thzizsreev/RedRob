#!/usr/bin/env python3
"""
Offline Phase 1 — INSTRUCTOR-large ONNX block-weighted vector precomputation.

GPU via ONNX Runtime (CUDA) only.

Example (AWS / local):
    python precompute.py --candidates data/candidates.jsonl --output-dir artifacts/candidates

Outputs (under --output-dir):
  candidate_index.faiss  (IndexFlatIP, 2304-d)
  id_map.json
  jd_query_vec.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from tracks.instructor.index import build_vector_index, build_vector_index_from_records
from tracks.instructor.onnx_embedder import load_embedder, unload_embedder
from tracks.shared.paths import ARTIFACTS_DIR, CANDIDATES_JSONL_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute INSTRUCTOR block-weighted candidate vectors and FAISS index.",
    )
    parser.add_argument(
        "--candidates",
        "-c",
        type=Path,
        default=CANDIDATES_JSONL_PATH,
        help=(
            "Candidate file: JSONL, JSONL.gz, or JSON array. "
            f"Default: {CANDIDATES_JSONL_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help=(
            "Directory for FAISS index and vectors. "
            "Default: artifacts/<candidates_stem> (e.g. artifacts/candidates)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of candidates (for smoke tests).",
    )
    return parser.parse_args(argv)


def resolve_output_dir(candidates_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    stem = candidates_path.name
    for suffix in (".jsonl.gz", ".jsonl", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = candidates_path.stem
    return (ARTIFACTS_DIR / stem).resolve()


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
    *,
    limit: int | None = None,
) -> Path:
    candidates_path = candidates_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Candidates:  {candidates_path}")
    print(f"Output dir:  {output_dir}")

    if candidates_path.suffix == ".json":
        candidates = load_candidates_json(candidates_path)
        print(f"Processing candidates from {candidates_path}...")
        build_vector_index_from_records(
            candidates,
            model,
            output_dir,
            limit=limit,
        )
    else:
        build_vector_index(
            candidates_path,
            model,
            output_dir,
            limit=limit,
        )

    return output_dir


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def print_output_manifest(output_dir: Path) -> None:
    artifacts = [
        output_dir / "candidate_index.faiss",
        output_dir / "id_map.json",
        output_dir / "jd_query_vec.npy",
    ]
    print("\n--- Output files ---")
    for path in artifacts:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path}")
    print(f"\nUse this directory for stage1.py / rank.py ARTIFACTS_PATH:\n  {output_dir}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidates_path = args.candidates.resolve()
    output_dir = resolve_output_dir(candidates_path, args.output_dir)

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    model = load_embedder()
    started = perf_counter()
    try:
        run_precompute(
            candidates_path,
            model,
            output_dir,
            limit=args.limit,
        )
    finally:
        unload_embedder(model)

    elapsed = perf_counter() - started
    print_output_manifest(output_dir)
    print(f"\nPrecompute complete in {_format_duration(elapsed)}.")


if __name__ == "__main__":
    main()
