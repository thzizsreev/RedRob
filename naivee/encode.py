"""BGE encoding for passages and JD queries."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_DIM, MODEL_NAME


def load_encoder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"      Model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    print(f"      Embedding dimension: {dim}")
    return model


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def encode_passages(model: SentenceTransformer, passages: list[str]) -> np.ndarray:
    print(f"      Prefixing {len(passages)} passages with 'passage:' ...")
    prefixed = [f"passage: {text}" for text in passages]
    print(f"      Running batch encode (show_progress_bar=True)...")
    vecs = model.encode(prefixed, normalize_embeddings=False, show_progress_bar=True)
    print(f"      L2-normalizing {len(vecs)} vectors to dim={EMBEDDING_DIM}...")
    return np.array([_l2_normalize(v) for v in vecs], dtype=np.float32)


def encode_query(model: SentenceTransformer, query_text: str) -> np.ndarray:
    print(f"      Prefixing query with 'query:' ...")
    vec = model.encode(f"query: {query_text}", normalize_embeddings=False)
    normalized = _l2_normalize(vec)
    print(f"      Query vector norm (should be ~1.0): {np.linalg.norm(normalized):.4f}")
    return normalized
