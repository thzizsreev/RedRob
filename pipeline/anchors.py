"""Stage 0: JD-derived anchor vector construction."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    ANCHORS_DIR,
    RETRIEVAL_ANCHOR_SENTENCES,
    INFRA_ANCHOR_SENTENCES,
    EVAL_ANCHOR_SENTENCES,
    
)
from pipeline.model_utils import embedding_dim

from pipeline.config import MODEL_NAME



def build_anchor(sentences: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode JD requirement sentences with query: prefix, mean, L2 normalize."""
    print(f"Building anchor for {sentences}")
    prefixed = sentences
    vecs = model.encode(prefixed, normalize_embeddings=False)
    anchor = np.mean(vecs, axis=0)
    norm = np.linalg.norm(anchor)
    if norm == 0:
        return np.zeros(embedding_dim(model), dtype=np.float32)
    return (anchor / norm).astype(np.float32)


def build_anchors(model: SentenceTransformer) -> dict[str, np.ndarray]:
    return {
        "retrieval": build_anchor(RETRIEVAL_ANCHOR_SENTENCES, model),
        "infra": build_anchor(INFRA_ANCHOR_SENTENCES, model),
        "eval": build_anchor(EVAL_ANCHOR_SENTENCES, model),
    }


def save_anchors(anchors: dict[str, np.ndarray], output_dir=ANCHORS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "anchor_retrieval.npy", anchors["retrieval"])
    np.save(output_dir / "anchor_infra.npy", anchors["infra"])
    np.save(output_dir / "anchor_eval.npy", anchors["eval"])


def load_anchors(input_dir=ANCHORS_DIR) -> dict[str, np.ndarray]:
    return {
        "retrieval": np.load(input_dir / "anchor_retrieval.npy"),
        "infra": np.load(input_dir / "anchor_infra.npy"),
        "eval": np.load(input_dir / "anchor_eval.npy"),
    }


def anchors_exist(input_dir=ANCHORS_DIR) -> bool:
    return all(
        (input_dir / name).exists()
        for name in ("anchor_retrieval.npy", "anchor_infra.npy", "anchor_eval.npy")
    )


def ensure_anchors(
    model: SentenceTransformer,
    *,
    rebuild: bool = False,
    output_dir=ANCHORS_DIR,
) -> dict[str, np.ndarray]:
    if not rebuild and anchors_exist(output_dir):
        print(f"Loading existing anchors from {output_dir}")
        return load_anchors(output_dir)

    print("Building anchor vectors from JD requirement sentences...")
    anchors = build_anchors(model)
    save_anchors(anchors, output_dir)
    for name, vec in anchors.items():
        print(f"  anchor_{name}: shape={vec.shape}")
    return anchors
    
if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    anchors = ensure_anchors(model, rebuild=True)
    #print(anchors)
    # print(anchors["retrieval"].shape)
    # print(anchors["infra"].shape)
    # print(anchors["eval"].shape)