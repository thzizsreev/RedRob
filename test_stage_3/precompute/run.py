#!/usr/bin/env python3
"""Stage 3 precompute — cohort fixtures, query vectors, survivor indices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_stage_3.precompute.cohort import (
    build_cohort,
    build_survivor_indices,
    write_stage0_pointer,
)
from test_stage_3.precompute.manifest import cohort_config_hash, query_config_hash
from test_stage_3.precompute.query_vectors import build_query_vectors
from test_stage_3.shared.config_precompute import DEFAULT_CONFIG, load_precompute_config
from test_stage_3.shared.io_precompute import (
    PrecomputeManifest,
    load_manifest,
    utc_now_iso,
    write_manifest,
)

PRECOMPUTE_DIR = Path(__file__).resolve().parent


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 precompute (one-time per cohort/JD).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite artifacts even if manifest hashes match",
    )
    return parser.parse_args()


def _should_skip(config_path: Path, force: bool) -> bool:
    if force:
        return False
    config = load_precompute_config(config_path)
    manifest_path = config.artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    existing = load_manifest(manifest_path)
    q_hash = query_config_hash(config)
    c_hash = cohort_config_hash(config)
    if existing.query_config_hash == q_hash and existing.cohort_config_hash == c_hash:
        print(f"Precompute up to date ({manifest_path}). Use --force to rebuild.")
        return True
    return False


def run_precompute(config_path: Path | None = None) -> Path:
    config = load_precompute_config(config_path)
    artifacts_dir = config.artifacts_dir
    cohort_dir = artifacts_dir / "cohort"
    query_vectors_dir = artifacts_dir / "query_vectors"

    stage2_df = build_cohort(config, cohort_dir)
    build_survivor_indices(config, stage2_df, cohort_dir)
    stage0_pointer = write_stage0_pointer(config, artifacts_dir)
    build_query_vectors(config, query_vectors_dir)

    manifest = PrecomputeManifest(
        version=1,
        created_at=utc_now_iso(),
        query_config_hash=query_config_hash(config),
        cohort_config_hash=cohort_config_hash(config),
        cohort_row_count=stage2_df.height,
        paths={
            "query_vectors_dir": _rel_or_abs(query_vectors_dir),
            "cohort_stage2": _rel_or_abs(cohort_dir / "stage2_gated.parquet"),
            "cohort_features": _rel_or_abs(cohort_dir / "candidate_features.parquet"),
            "survivor_row_indices": _rel_or_abs(cohort_dir / "survivor_row_indices.npy"),
            "stage0_pointer": _rel_or_abs(stage0_pointer),
        },
    )
    manifest_path = artifacts_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path}")
    return manifest_path


def main() -> None:
    args = parse_args()
    if _should_skip(args.config, args.force):
        return
    run_precompute(args.config)
    print("\nPrecompute complete. Run: python test_stage_3/runner/run.py")


if __name__ == "__main__":
    main()
