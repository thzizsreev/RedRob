"""Stage 0: JD-derived anchor vector construction."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    ANCHORS_DIR,
    JD_EVAL_SENTENCES,
    JD_INFRA_SENTENCES,
    JD_RETRIEVAL_SENTENCES,
)
from pipeline.model_utils import embedding_dim


def build_anchor(sentences: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode JD requirement sentences with query: prefix, mean, L2 normalize."""
    prefixed = [f"query: {s}" for s in sentences]
    vecs = model.encode(prefixed, normalize_embeddings=False)
    anchor = np.mean(vecs, axis=0)
    norm = np.linalg.norm(anchor)
    if norm == 0:
        return np.zeros(embedding_dim(model), dtype=np.float32)
    return (anchor / norm).astype(np.float32)


def build_anchors(model: SentenceTransformer) -> dict[str, np.ndarray]:
    return {
        "retrieval": build_anchor(JD_RETRIEVAL_SENTENCES, model),
        "infra": build_anchor(JD_INFRA_SENTENCES, model),
        "eval": build_anchor(JD_EVAL_SENTENCES, model),
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
