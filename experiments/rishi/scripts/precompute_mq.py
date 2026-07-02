#!/usr/bin/env python3
"""Precompute multi-query JD facet similarity scores (TRACER offline step 2)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranker import jd_config as jd
from ranker.embeddings import _encode_dense, _load_sentence_model, candidate_to_text
from ranker.io import iter_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-query JD facet scores for TRACER.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    art = Path(args.artifacts)
    ids_path = art / "candidate_ids.json"
    if not ids_path.exists():
        print("Run precompute_embeddings.py first.", file=sys.stderr)
        return 1

    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    id_set = set(ids)

    id_to_text: dict[str, str] = {}
    for candidate in iter_candidates(args.candidates):
        cid = candidate["candidate_id"]
        if cid in id_set:
            id_to_text[cid] = candidate_to_text(candidate)

    texts = [id_to_text[cid] for cid in ids if cid in id_to_text]
    if len(texts) != len(ids):
        print(f"Warning: {len(texts)} texts for {len(ids)} ids", file=sys.stderr)

    model = _load_sentence_model(jd.EMBED_MODEL, device=args.device)
    facet_vecs = model.encode(jd.JD_FACETS, normalize_embeddings=True).astype(np.float32)
    cand_vecs = _encode_dense(model, texts, args.batch_size)

    # max-pool facet similarities per candidate
    sims = cand_vecs @ facet_vecs.T
    mq = np.clip((sims.max(axis=1) + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

    np.save(art / "mq_scores.npy", mq)

    hybrid_path = art / "hybrid_scores.npy"
    if not hybrid_path.exists():
        hybrid_path = art / "semantic_scores.npy"
    hybrid = np.load(hybrid_path).astype(np.float32)

    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    meta["tracer_mq"] = True
    meta["mq_facets"] = len(jd.JD_FACETS)
    (art / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved mq_scores.npy ({len(mq)} scores) to {art}")
    print(f"MQ range: {mq.min():.3f} - {mq.max():.3f}; hybrid range: {hybrid.min():.3f} - {hybrid.max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
