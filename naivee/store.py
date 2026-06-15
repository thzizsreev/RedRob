"""FAISS index persistence for naive resume vectors."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from config import (
    ARTIFACTS_DIR,
    EMBEDDING_DIM,
    ID_MAP_FILENAME,
    INDEX_FILENAME,
    PASSAGES_FILENAME,
)


def save_index(
    vectors: np.ndarray,
    passages: list[tuple[str, str]],
    output_dir: Path = ARTIFACTS_DIR,
) -> faiss.Index:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"      Creating IndexFlatIP(dim={EMBEDDING_DIM})...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors.astype(np.float32))
    print(f"      Added {index.ntotal} vectors to index")

    id_map = {str(i): candidate_id for i, (candidate_id, _) in enumerate(passages)}

    index_path = output_dir / INDEX_FILENAME
    id_map_path = output_dir / ID_MAP_FILENAME
    passages_path = output_dir / PASSAGES_FILENAME

    print(f"      Writing {index_path.name}...")
    faiss.write_index(index, str(index_path))

    print(f"      Writing {id_map_path.name} ({len(id_map)} entries)...")
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    print(f"      Writing {passages_path.name}...")
    with open(passages_path, "w", encoding="utf-8") as f:
        for candidate_id, passage in passages:
            f.write(json.dumps({"candidate_id": candidate_id, "passage": passage}) + "\n")

    print(f"      Done — index: {index.ntotal} vectors, dim={EMBEDDING_DIM}")

    return index


def load_index(
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[faiss.Index, dict[int, str]]:
    index_path = artifacts_dir / INDEX_FILENAME
    id_map_path = artifacts_dir / ID_MAP_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}. Run naivee/precompute.py first.")
    if not id_map_path.exists():
        raise FileNotFoundError(f"ID map not found: {id_map_path}. Run naivee/precompute.py first.")

    print(f"      Reading {index_path.name}...")
    index = faiss.read_index(str(index_path))
    print(f"      Index loaded: {index.ntotal} vectors, dim={index.d}")

    print(f"      Reading {id_map_path.name}...")
    with open(id_map_path, encoding="utf-8") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}
    print(f"      ID map loaded: {len(id_map)} candidate IDs")

    return index, id_map
