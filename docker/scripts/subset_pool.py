#!/usr/bin/env python3
"""Filter baked pool1k artifacts to an active candidate ID set (≤100)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import faiss
import numpy as np
import polars as pl

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.core.config import (
    CANDIDATE_FEATURES_FILENAME,
    CANDIDATE_VECTORS_FILENAME,
    ID_MAP_FILENAME,
    INDEX_FILENAME,
    JD_QUERY_VEC_FILENAME,
    STAGE1_CLUSTER_LABELS_FILENAME,
    STAGE1_CLUSTER_MANIFEST_FILENAME,
    STAGE1_UMAP_REDUCED_FILENAME,
    STAGE3_QUERY_MANIFEST_FILENAME,
    STAGE3_QUERY_VECTORS_DIR,
    VECTOR_DIM,
)
from tracks.instructor.core.io import load_index_and_id_map


def _resolve_indices(
    source_stage0: Path,
    active_ids: list[str],
) -> tuple[list[int], list[str]]:
    _, id_map = load_index_and_id_map(source_stage0)
    row_to_id = [id_map[i] for i in range(len(id_map))]
    id_to_row = {cid: i for i, cid in enumerate(row_to_id)}

    missing = [cid for cid in active_ids if cid not in id_to_row]
    if missing:
        raise ValueError(
            f"{len(missing)} active IDs not in baked pool: {missing[:5]}"
        )

    indices = [id_to_row[cid] for cid in active_ids]
    ordered_ids = active_ids
    return indices, ordered_ids


def _reconstruct_vectors(index: faiss.Index, indices: list[int]) -> np.ndarray:
    """Fetch aligned vectors via FAISS reconstruct + id_map row indices."""
    return np.vstack([index.reconstruct(i) for i in indices]).astype(np.float32)


def _write_faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vectors.astype(np.float32))
    return index


def subset_pool(
    source_stage0: Path,
    source_stage1: Path,
    target_stage0: Path,
    target_stage1: Path,
    active_ids: list[str],
) -> None:
    indices, ordered_ids = _resolve_indices(source_stage0, active_ids)
    n = len(indices)

    target_stage0.mkdir(parents=True, exist_ok=True)
    target_stage1.mkdir(parents=True, exist_ok=True)

    index, _ = load_index_and_id_map(source_stage0)
    subset_vectors = _reconstruct_vectors(index, indices)

    np.save(target_stage0 / CANDIDATE_VECTORS_FILENAME, subset_vectors)

    subset_index = _write_faiss_index(subset_vectors)
    faiss.write_index(subset_index, str(target_stage0 / INDEX_FILENAME))

    new_id_map = {str(i): cid for i, cid in enumerate(ordered_ids)}
    with open(target_stage0 / ID_MAP_FILENAME, "w", encoding="utf-8") as f:
        json.dump(new_id_map, f)

    jd_src = source_stage0 / JD_QUERY_VEC_FILENAME
    if jd_src.exists():
        shutil.copy2(jd_src, target_stage0 / JD_QUERY_VEC_FILENAME)

    features_src = source_stage0 / CANDIDATE_FEATURES_FILENAME
    if features_src.exists():
        df = pl.read_parquet(features_src)
        subset_df = df.filter(pl.col("candidate_id").is_in(ordered_ids))
        subset_df.write_parquet(target_stage0 / CANDIDATE_FEATURES_FILENAME)

    query_dir_name = STAGE3_QUERY_VECTORS_DIR
    src_query = source_stage0 / query_dir_name
    dst_query = target_stage0 / query_dir_name
    if src_query.exists():
        if dst_query.exists():
            shutil.rmtree(dst_query, ignore_errors=True)
        dst_query.mkdir(parents=True, exist_ok=True)
        for item in src_query.iterdir():
            dest_item = dst_query / item.name
            if item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest_item)

    manifest_src = source_stage0 / STAGE3_QUERY_MANIFEST_FILENAME
    if manifest_src.exists():
        manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
        manifest["query_vectors_dir"] = query_dir_name
        (target_stage0 / STAGE3_QUERY_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    for fname in ("precompute_manifest.json", "manifest.json"):
        src = source_stage0 / fname
        if src.exists():
            shutil.copy2(src, target_stage0 / fname)

    _subset_stage1_cluster(source_stage1, target_stage1, indices, n)


def _subset_stage1_cluster(
    source_stage1: Path,
    target_stage1: Path,
    indices: list[int],
    n: int,
) -> None:
    vectors = np.load(source_stage1 / CANDIDATE_VECTORS_FILENAME).astype(np.float32)
    labels = np.load(source_stage1 / STAGE1_CLUSTER_LABELS_FILENAME)
    reduced = np.load(source_stage1 / STAGE1_UMAP_REDUCED_FILENAME).astype(np.float32)

    np.save(target_stage1 / CANDIDATE_VECTORS_FILENAME, vectors[indices])
    np.save(target_stage1 / STAGE1_CLUSTER_LABELS_FILENAME, labels[indices])
    np.save(target_stage1 / STAGE1_UMAP_REDUCED_FILENAME, reduced[indices])

    manifest_path = source_stage1 / STAGE1_CLUSTER_MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["n_candidates"] = n
        (target_stage1 / STAGE1_CLUSTER_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset pool1k artifacts to active IDs.")
    parser.add_argument("--source-stage0", type=Path, required=True)
    parser.add_argument("--source-stage1", type=Path, required=True)
    parser.add_argument("--target-stage0", type=Path, required=True)
    parser.add_argument("--target-stage1", type=Path, required=True)
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Active candidate IDs (or use --ids-file)",
    )
    parser.add_argument("--ids-file", type=Path, help="JSON file with candidate_ids list")
    args = parser.parse_args()

    if args.ids_file:
        payload = json.loads(args.ids_file.read_text(encoding="utf-8"))
        active_ids = payload if isinstance(payload, list) else payload["candidate_ids"]
    elif args.ids:
        active_ids = args.ids
    else:
        raise ValueError("Provide --ids or --ids-file")

    subset_pool(
        args.source_stage0,
        args.source_stage1,
        args.target_stage0,
        args.target_stage1,
        active_ids,
    )
    print(f"Subset {len(active_ids)} candidates → {args.target_stage0}")


if __name__ == "__main__":
    main()
