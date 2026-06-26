"""Stage 3 I/O — load assets, validate schema, write outputs."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import polars as pl

from tracks.instructor.core.config import (
    CANDIDATE_FEATURES_FILENAME,
    CANDIDATE_VECTORS_FILENAME,
    STAGE3_QUERY_MANIFEST_FILENAME,
    VECTOR_DIM,
)
from tracks.instructor.core.io import load_index_and_id_map, load_vectors_from_artifacts
from tracks.instructor.stage3.config import Stage3Config

REQUIRED_STAGE2_COLUMNS = frozenset(
    {
        "candidate_id",
        "cluster_id",
        "dist_to_centroid",
        "exp_band",
        "in_sweet_spot",
        "title_family",
        "skill_kw_density",
        "title_ambiguous",
        "stale_profile",
        "low_responder",
        "not_open",
    }
)


@dataclass(frozen=True)
class RetrievalAssets:
    index: faiss.Index
    vectors: np.ndarray
    id_to_row: dict[str, int]
    row_to_id: list[str]


def load_stage2_gated(path: Path, config: Stage3Config) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 output: {path}")

    df = pl.read_parquet(path)
    missing = REQUIRED_STAGE2_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"stage2_gated.parquet missing required columns: {sorted(missing)}")

    count = df.height
    print(f"Stage 2 survivors loaded: {count:,}")
    if count < config.expected_survivor_min or count > config.expected_survivor_max:
        warnings.warn(
            f"Stage 2 survivor count {count} outside expected range "
            f"[{config.expected_survivor_min}, {config.expected_survivor_max}]",
            stacklevel=2,
        )
    return df


def load_retrieval_assets(artifacts_path: Path) -> RetrievalAssets:
    index, id_map = load_index_and_id_map(artifacts_path)

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(
            f"FAISS index must use METRIC_INNER_PRODUCT, got {index.metric_type}"
        )

    row_to_id = [id_map[i] for i in range(index.ntotal)]
    id_to_row = {cid: i for i, cid in enumerate(row_to_id)}

    vectors_path = artifacts_path / CANDIDATE_VECTORS_FILENAME
    if vectors_path.exists():
        vectors = np.load(vectors_path).astype(np.float32)
        print(f"Loaded candidate vectors: {vectors.shape}")
    else:
        warnings.warn(
            f"{vectors_path} not found — reconstructing vectors from FAISS index",
            stacklevel=2,
        )
        _, vectors = load_vectors_from_artifacts(artifacts_path)

    if vectors.shape != (index.ntotal, VECTOR_DIM):
        raise ValueError(
            f"Vector shape {vectors.shape} does not match index "
            f"({index.ntotal}, {VECTOR_DIM})"
        )

    print(f"FAISS index: {index.ntotal:,} vectors, dim={index.d}")
    return RetrievalAssets(
        index=index,
        vectors=vectors,
        id_to_row=id_to_row,
        row_to_id=row_to_id,
    )


def load_skill_features(artifacts_path: Path) -> pl.DataFrame:
    path = artifacts_path / CANDIDATE_FEATURES_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing candidate features: {path}. "
            "Re-run tracks/instructor/stage0/run.py to build skill_weighted_score"
        )

    df = pl.read_parquet(path, columns=["candidate_id", "skill_weighted_score"])
    if "skill_weighted_score" not in df.columns:
        raise ValueError(
            "skill_weighted_score column not found in candidate_features.parquet — "
            "ensure Stage 0 skill precompute has been run."
        )
    print(f"Loaded skill features: {df.height:,} candidates")
    return df


def load_query_vectors(query_vectors_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = np.load(query_vectors_dir / "q1_vec.npy").astype(np.float32)
    q2 = np.load(query_vectors_dir / "q2_vec.npy").astype(np.float32)
    q3 = np.load(query_vectors_dir / "q3_vec.npy").astype(np.float32)
    return q1, q2, q3


def build_survivor_row_indices(
    stage2_df: pl.DataFrame,
    id_to_row: dict[str, int],
) -> np.ndarray:
    unknown: list[str] = []
    indices: list[int] = []
    for cid in stage2_df["candidate_id"].to_list():
        row = id_to_row.get(str(cid))
        if row is None:
            unknown.append(str(cid))
        else:
            indices.append(row)

    if unknown:
        examples = unknown[:5]
        raise ValueError(
            f"{len(unknown)} Stage 2 survivor(s) not found in id_map. Examples: {examples}"
        )

    return np.array(indices, dtype=np.int64)


def stage3_query_manifest_path(artifacts_path: Path) -> Path:
    return artifacts_path / STAGE3_QUERY_MANIFEST_FILENAME


def _write_retrieved_json(path: Path, df: pl.DataFrame) -> None:
    records = df.to_dicts() if df.height > 0 else []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def write_stage3_outputs(
    output_dir: Path,
    retrieved_df: pl.DataFrame,
    distribution_df: pl.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieved_df.write_parquet(output_dir / "stage3_retrieved.parquet")
    _write_retrieved_json(output_dir / "stage3_retrieved.json", retrieved_df)

    distribution_df.write_csv(output_dir / "stage3_score_distribution.csv")

    with open(output_dir / "stage3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {output_dir / 'stage3_retrieved.parquet'}")
    print(f"Wrote {output_dir / 'stage3_retrieved.json'}")
    print(f"Wrote {output_dir / 'stage3_score_distribution.csv'}")
    print(f"Wrote {output_dir / 'stage3_summary.json'}")
