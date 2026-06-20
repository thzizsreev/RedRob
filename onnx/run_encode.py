#!/usr/bin/env python3
"""Encode sample texts with INSTRUCTOR-large via ONNX."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.instructor.onnx_embedder import load_embedder, unload_embedder

PAIRS = [
    [
        "Represent the Science title: ",
        "3D ActionSLAM: wearable person tracking in multi-floor environments",
    ],
    [
        "Represent the Wikipedia document for retrieval: ",
        "Artificial intelligence was founded as an academic discipline in 1956.",
    ],
    [
        "Represent the AI engineering career history for retrieving candidates "
        "with production experience in semantic search and embeddings-based retrieval: ",
        "Built FAISS indexes and hybrid search pipelines serving millions of queries.",
    ],
]


def main() -> None:
    model = load_embedder()
    try:
        embeddings = model.encode(PAIRS, batch_size=8, normalize=True)
        print(f"shape: {embeddings.shape}")
        for i, pair in enumerate(PAIRS):
            preview = pair[1][:50] + ("..." if len(pair[1]) > 50 else "")
            print(f"[{i}] {preview!r}")
            print(f"     first 8 dims: {embeddings[i, :8]}")
    finally:
        unload_embedder(model)


if __name__ == "__main__":
    main()
