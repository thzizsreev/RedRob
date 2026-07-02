#!/usr/bin/env python3
"""
Fetch pool1k docker artifacts by candidate ID from full-pool precompute.

Uses id_map row indices + FAISS reconstruct — no ONNX encode, no re-cluster.
Requires artifacts/runtime/stage0 + stage1 from production Stage 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from docker.paths import MANIFEST_PATH, POOL1K_JSONL, POOL1K_STAGE0, POOL1K_STAGE1
from docker.scripts.subset_pool import subset_pool
from tracks.instructor.core.io import iter_candidates_from_path
from tracks.shared.paths import RUNTIME_STAGE0_DIR, RUNTIME_STAGE1_DIR


def _load_pool_ids(candidates_path: Path) -> list[str]:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return list(manifest["candidate_ids"])
    return [str(r["candidate_id"]) for r in iter_candidates_from_path(candidates_path)]


def _update_manifest(pool_ids: list[str], source_stage0: Path) -> None:
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.update(
        {
            "n_pool": len(pool_ids),
            "candidate_ids": pool_ids,
            "precompute_method": "id_map_faiss_reconstruct",
            "source_stage0": str(source_stage0.resolve()),
            "artifacts_fetched_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        }
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch pool1k artifacts by ID from full-pool stage0/stage1."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=POOL1K_JSONL,
        help="pool1k.jsonl (IDs must exist in source id_map)",
    )
    parser.add_argument(
        "--source-stage0",
        type=Path,
        default=RUNTIME_STAGE0_DIR,
        help="Full-pool stage0 (FAISS + id_map)",
    )
    parser.add_argument(
        "--source-stage1",
        type=Path,
        default=RUNTIME_STAGE1_DIR,
        help="Full-pool stage1 cluster npy",
    )
    parser.add_argument("--stage0-out", type=Path, default=POOL1K_STAGE0)
    parser.add_argument("--stage1-out", type=Path, default=POOL1K_STAGE1)
    args = parser.parse_args()

    if not args.candidates.exists():
        raise FileNotFoundError(f"Candidates not found: {args.candidates}")
    if not (args.source_stage0 / "candidate_index.faiss").exists():
        raise FileNotFoundError(
            f"Full-pool index missing at {args.source_stage0}. "
            "Run tracks/instructor/stage0/run.py on the full pool first."
        )
    if not (args.source_stage1 / "cluster_labels.npy").exists():
        raise FileNotFoundError(
            f"Full-pool cluster artifacts missing at {args.source_stage1}. "
            "Run tracks/instructor/stage0/run_cluster.py first."
        )

    pool_ids = _load_pool_ids(args.candidates)
    if len(pool_ids) != 1000:
        print(f"Warning: expected 1000 pool IDs, got {len(pool_ids)}", file=sys.stderr)

    started = perf_counter()
    subset_pool(
        args.source_stage0,
        args.source_stage1,
        args.stage0_out,
        args.stage1_out,
        pool_ids,
    )
    _update_manifest(pool_ids, args.source_stage0)

    elapsed = perf_counter() - started
    print(f"Fetched {len(pool_ids)} candidates in {elapsed:.1f}s")
    print(f"  stage0: {args.stage0_out}")
    print(f"  stage1: {args.stage1_out}")
    print(f"  manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
