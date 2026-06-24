"""BM25 index precompute — corpus order matches FAISS id_map row order."""

from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from tracks.instructor.core.config import BM25_INDEX_FILENAME
from tracks.instructor.stage3.jargon import build_jargon_text, tokenize_jargon


def build_bm25_index(
    records: list[dict],
    output_dir: Path,
    *,
    q4_tokens: list[str] | None = None,
) -> BM25Okapi:
    """Build and save BM25Okapi over jargon text, one document per record in order."""
    token_set = frozenset(t.lower() for t in q4_tokens) if q4_tokens else None
    corpus = [
        tokenize_jargon(build_jargon_text(record, q4_tokens=token_set))
        for record in records
    ]
    bm25 = BM25Okapi(corpus)
    output_path = output_dir / BM25_INDEX_FILENAME
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index built: {len(corpus):,} documents")
    print(f"Saved {output_path}")
    return bm25
