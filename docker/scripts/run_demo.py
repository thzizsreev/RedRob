#!/usr/bin/env python3
"""Docker sandbox — rank full 1K pool, output top-100 ranking CSV."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from docker.paths import (
    ACTIVE_JSONL,
    DEFAULT_OUTPUT_CONTAINER,
    MANIFEST_PATH,
    POOL1K_JSONL,
    PRECOMPUTED_POOL1K,
    ROOT_DIR,
    SANDBOX_CONFIG,
    WORK_CONFIG,
    WORK_DATA,
    WORK_RUNTIME,
)
from docker.scripts.make_sandbox_config import POOL_SIZE, SUBMISSION_TOP_N, write_config
from docker.scripts.subset_pool import subset_pool
from tracks.instructor.core.config import (
    STAGE1_CANDIDATE_VECTORS_FILENAME,
    STAGE1_CLUSTER_LABELS_FILENAME,
    STAGE1_CLUSTER_MANIFEST_FILENAME,
    STAGE1_UMAP_REDUCED_FILENAME,
)
from tracks.instructor.core.io import iter_candidates_from_path
from tracks.instructor.pipeline import RankingPipelineConfig, run_ranking_pipeline
from tracks.instructor.stage5.io import write_ranking_csv, write_submission_csv
from tracks.instructor.stage5.validate import validate_ranking_csv

MAX_UPLOAD = 100
TIME_BUDGET_SECONDS = 300


def _resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return args.input
    env_input = os.environ.get("REDROB_INPUT")
    if env_input:
        return Path(env_input)
    container_input = Path("/input/candidates.jsonl")
    if container_input.exists():
        return container_input
    return POOL1K_JSONL


def _is_upload_input(args: argparse.Namespace, input_path: Path) -> bool:
    if args.input is not None:
        return True
    if os.environ.get("REDROB_INPUT"):
        return True
    return Path("/input/candidates.jsonl").exists() and input_path.exists()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    env_out = os.environ.get("REDROB_OUTPUT")
    if env_out:
        return Path(env_out)
    if DEFAULT_OUTPUT_CONTAINER.exists() or os.environ.get("REDROB_IN_CONTAINER"):
        return DEFAULT_OUTPUT_CONTAINER
    from tracks.shared.paths import RANKING_CSV_PATH

    return RANKING_CSV_PATH.parent


def _load_pool_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. Run docker/scripts/sample_pool1k.py first."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _validate_upload_subset(
    records: list[dict],
    pool_ids: set[str],
) -> tuple[list[dict], list[str]]:
    if not records:
        raise ValueError("No candidates in upload input")
    if len(records) > MAX_UPLOAD:
        raise ValueError(
            f"Upload has {len(records)} candidates; at most {MAX_UPLOAD} per run."
        )

    ids: list[str] = []
    for record in records:
        cid = str(record.get("candidate_id", ""))
        if cid not in pool_ids:
            raise ValueError(
                f"Candidate {cid} is not in the pool1k manifest. "
                "Upload only IDs from the sandbox pool."
            )
        ids.append(cid)

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate_id values in upload")

    return records, ids


def _write_active_jsonl(records: list[dict]) -> None:
    WORK_DATA.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_JSONL, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _copy_stage1_cluster(source: Path, dest: Path) -> None:
    """Copy read-only cluster npy into work stage1 (filter writes JSON alongside)."""
    _reset_dir(dest)
    for name in (
        STAGE1_CANDIDATE_VECTORS_FILENAME,
        STAGE1_CLUSTER_LABELS_FILENAME,
        STAGE1_UMAP_REDUCED_FILENAME,
        STAGE1_CLUSTER_MANIFEST_FILENAME,
    ):
        shutil.copy2(source / name, dest / name)


def _stage_output_dirs() -> dict[str, Path]:
    return {
        "stage2_output_dir": WORK_RUNTIME / "stage2",
        "stage3_output_dir": WORK_RUNTIME / "stage3",
        "stage4_output_dir": WORK_RUNTIME / "stage4",
        "stage5_output_dir": WORK_RUNTIME / "stage5",
    }


def _maybe_fetch_pool_artifacts() -> None:
    if os.environ.get("REDROB_FETCH_POOL") != "1":
        return
    script = ROOT_DIR / "docker" / "scripts" / "fetch_pool1k_artifacts.py"
    subprocess.run([sys.executable, str(script)], check=True)


def _pad_submission_csv(csv_path: Path, pool_ids: list[str]) -> None:
    """Upload-only fallback when gates leave fewer than 100 scored rows."""
    import csv

    rows: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) >= SUBMISSION_TOP_N:
        return

    present = {r["candidate_id"] for r in rows}
    last_score = float(rows[-1]["score"]) if rows else 0.5
    missing = sorted(cid for cid in pool_ids if cid not in present)

    padded = list(rows)
    rank = len(rows) + 1
    for i, cid in enumerate(missing, start=1):
        score = round(last_score - 0.001 * i, 6)
        padded.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "score": score,
                "reasoning": (
                    "Did not pass early screening gates; ranked below qualified candidates."
                ),
            }
        )
        rank += 1
        if len(padded) >= SUBMISSION_TOP_N:
            break

    write_submission_csv(csv_path, padded[:SUBMISSION_TOP_N])


def _pad_ranking_csv(csv_path: Path, pool_ids: list[str]) -> None:
    """Upload-only fallback when gates leave fewer than 100 scored rows."""
    import csv

    rows: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) >= SUBMISSION_TOP_N:
        return

    present = {r["candidate_id"] for r in rows}
    last_score = float(rows[-1]["score"]) if rows else 0.5
    missing = sorted(cid for cid in pool_ids if cid not in present)

    padded = list(rows)
    rank = len(rows) + 1
    for i, cid in enumerate(missing, start=1):
        score = round(last_score - 0.001 * i, 6)
        padded.append({"candidate_id": cid, "rank": rank, "score": score})
        rank += 1
        if len(padded) >= SUBMISSION_TOP_N:
            break

    write_ranking_csv(csv_path, padded[:SUBMISSION_TOP_N])


def _validate_ranking(csv_path: Path) -> None:
    try:
        validate_ranking_csv(csv_path, expected_rows=SUBMISSION_TOP_N)
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("Ranking CSV is valid.")


def _validate_submission(csv_path: Path) -> None:
    validator = ROOT_DIR / "tools" / "validate_submission.py"
    result = subprocess.run(
        [sys.executable, str(validator), str(csv_path)],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sandbox: rank full 1K pool, write top-100 CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional upload JSONL (<=100 IDs, subset of pool1k)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--precomputed",
        type=Path,
        default=PRECOMPUTED_POOL1K,
        help="Baked pool1k artifacts root",
    )
    args = parser.parse_args()

    os.environ["REDROB_CPU_ONLY"] = "1"
    started = perf_counter()

    input_path = _resolve_input_path(args)
    upload_mode = _is_upload_input(args, input_path)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    _maybe_fetch_pool_artifacts()

    manifest = _load_pool_manifest()
    pool_ids_list = list(manifest["candidate_ids"])
    pool_ids = set(pool_ids_list)
    n_pool = int(manifest.get("n_pool", len(pool_ids_list)))

    source_stage0 = args.precomputed / "stage0"
    source_stage1 = args.precomputed / "stage1"
    if not (source_stage0 / "candidate_index.faiss").exists():
        raise FileNotFoundError(
            f"Missing pool1k artifacts at {source_stage0}. "
            "Run docker/scripts/fetch_pool1k_artifacts.py."
        )

    outputs = _stage_output_dirs()
    for d in outputs.values():
        _reset_dir(d)
    if upload_mode:
        if not input_path.exists():
            raise FileNotFoundError(f"Upload input not found: {input_path}")
        records = list(iter_candidates_from_path(input_path))
        records, active_ids = _validate_upload_subset(records, pool_ids)
        _write_active_jsonl(records)
        work_stage0 = WORK_RUNTIME / "stage0"
        work_stage1 = WORK_RUNTIME / "stage1"
        _reset_dir(work_stage0)
        _reset_dir(work_stage1)
        subset_pool(
            source_stage0,
            source_stage1,
            work_stage0,
            work_stage1,
            active_ids,
        )
        stage0_path = work_stage0
        stage1_path = work_stage1
        candidates_path = ACTIVE_JSONL
        n_run = len(active_ids)
        full_pool = False
        print(f"\nUpload mode: ranking {n_run} candidates (subset of pool1k)...")
    else:
        if not POOL1K_JSONL.exists():
            raise FileNotFoundError(f"Missing {POOL1K_JSONL}")
        stage0_path = source_stage0
        work_stage1 = WORK_RUNTIME / "stage1"
        _copy_stage1_cluster(source_stage1, work_stage1)
        stage1_path = work_stage1
        candidates_path = POOL1K_JSONL
        n_run = n_pool
        full_pool = True
        active_ids = pool_ids_list
        print(f"\nFull pool mode: ranking all {n_run} candidates, top {SUBMISSION_TOP_N} CSV...")

    config_path = write_config(WORK_CONFIG, n_pool=n_run, full_pool=full_pool)
    if not SANDBOX_CONFIG.exists():
        write_config(SANDBOX_CONFIG, n_pool=POOL_SIZE, full_pool=True)

    pipeline_config = RankingPipelineConfig(
        stage0_path=stage0_path,
        stage1_path=stage1_path,
        candidates_path=candidates_path,
        config_path=config_path,
        stage1_floor=100,
        print_summaries=True,
        include_reasoning=False,
        **outputs,
    )

    result = run_ranking_pipeline(pipeline_config)

    from tracks.shared.paths import RANKING_CSV_PATH

    dest_csv = output_dir / RANKING_CSV_PATH.name
    shutil.copy2(result.final_csv_path, dest_csv)
    if upload_mode:
        _pad_ranking_csv(dest_csv, active_ids)
    print(f"\nRanking CSV (top {SUBMISSION_TOP_N}): {dest_csv}")

    _validate_ranking(dest_csv)

    elapsed = perf_counter() - started
    print(f"\nTotal sandbox time: {elapsed:.1f}s")
    if elapsed > TIME_BUDGET_SECONDS:
        print(
            f"WARNING: exceeded {TIME_BUDGET_SECONDS}s CPU budget ({elapsed:.1f}s)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
