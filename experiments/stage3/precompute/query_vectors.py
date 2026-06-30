"""Encode and persist Q1/Q2/Q3 query vectors."""

from __future__ import annotations

from pathlib import Path

from experiments.stage3.shared.config_precompute import PrecomputeConfig
from experiments.stage3.shared.cpu_embedder import load_embedder, unload_embedder
from experiments.stage3.shared.io_precompute import save_query_vectors
from experiments.stage3.shared.query_encode import encode_stage3_queries


def build_query_vectors(config: PrecomputeConfig, query_vectors_dir: Path) -> None:
    model = load_embedder()
    try:
        q1, q2, q3 = encode_stage3_queries(model, config)
    finally:
        unload_embedder(model)

    save_query_vectors(query_vectors_dir, q1, q2, q3)
    print(f"Wrote query vectors to {query_vectors_dir}")
