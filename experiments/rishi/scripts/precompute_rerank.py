#!/usr/bin/env python3
"""Cross-encoder rerank boost for TRACER coarse top-K (offline step 3)."""

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
from ranker.embeddings import candidate_to_text, _get_device, _resolve_local_snapshot, _ssl_env
from ranker.io import iter_candidates

RERANKER_FALLBACKS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]


def _load_cross_encoder(device: str = "auto"):
    from sentence_transformers import CrossEncoder

    _ssl_env()
    dev = _get_device(device)
    last_err: Exception | None = None
    for model_name in RERANKER_FALLBACKS:
        local = _resolve_local_snapshot(model_name)
        path = local or model_name
        try:
            model = CrossEncoder(path, device=dev)
            print(f"[rerank] loaded {model_name} on {dev}")
            return model, model_name
        except Exception as exc:
            print(f"[rerank] {model_name} failed: {exc}")
            last_err = exc
    raise RuntimeError(f"No cross-encoder available: {last_err}")


def _biencoder_rerank_fallback(
    art: Path,
    ids: list[str],
    top_indices: np.ndarray,
) -> tuple[dict[str, float], str]:
    """Fallback: rerank top-K using precomputed MiniLM embeddings (no download)."""
    emb_path = art / "candidate_embeddings.npy"
    job_path = art / "job_embedding.npy"
    if not emb_path.exists() or not job_path.exists():
        raise FileNotFoundError("Missing candidate_embeddings.npy or job_embedding.npy for bi-encoder fallback")

    embeddings = np.load(emb_path).astype(np.float32)
    job_vec = np.load(job_path).astype(np.float32)
    pair_ids = [ids[i] for i in top_indices]
    rows = np.array(top_indices, dtype=np.int64)
    sims = embeddings[rows] @ job_vec
    sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
    lo, hi = float(sims.min()), float(sims.max())
    norm = (sims - lo) / (hi - lo + 1e-9)
    boost = {cid: float(norm[i]) for i, cid in enumerate(pair_ids)}
    print(f"[rerank] bi-encoder fallback on {len(boost)} candidates")
    return boost, "bi-encoder/minilm-cached-embeddings"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-encoder rerank boost for TRACER top-K.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    art = Path(args.artifacts)
    ids = json.loads((art / "candidate_ids.json").read_text(encoding="utf-8"))
    hybrid_path = art / "hybrid_scores.npy"
    if not hybrid_path.exists():
        hybrid_path = art / "semantic_scores.npy"
    hybrid = np.load(hybrid_path).astype(np.float32)

    top_indices = np.argsort(-hybrid)[: args.top_k]
    top_ids = {ids[i] for i in top_indices}

    id_to_text: dict[str, str] = {}
    for candidate in iter_candidates(args.candidates):
        cid = candidate["candidate_id"]
        if cid in top_ids:
            id_to_text[cid] = candidate_to_text(candidate)

    pair_ids = [ids[i] for i in top_indices if ids[i] in id_to_text]
    pairs = [(jd.JD_TEXT, id_to_text[cid]) for cid in pair_ids]

    if not pairs:
        print("No candidates loaded for rerank.", file=sys.stderr)
        return 1

    used_model = ""
    try:
        model, used_model = _load_cross_encoder(args.device)
        raw_scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
        raw = np.array(raw_scores, dtype=np.float32)
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / (hi - lo + 1e-9)
        boost = {cid: float(norm[i]) for i, cid in enumerate(pair_ids)}
    except Exception as exc:
        print(f"[rerank] cross-encoder unavailable ({exc}), using bi-encoder fallback")
        boost, used_model = _biencoder_rerank_fallback(art, ids, top_indices)

    (art / "rerank_boost.json").write_text(json.dumps(boost), encoding="utf-8")

    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    meta["tracer_rerank"] = True
    meta["rerank_top_k"] = args.top_k
    meta["rerank_model"] = used_model
    (art / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved rerank_boost.json ({len(boost)} entries) to {art}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
