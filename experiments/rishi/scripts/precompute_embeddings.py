#!/usr/bin/env python3
"""Precompute hybrid semantic scores (BGE/MiniLM dense + TF-IDF sparse)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ranker.embeddings import SemanticIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute JD–candidate semantic scores.")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "candidates.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--backend",
        choices=["hybrid", "tfidf", "minilm"],
        default="hybrid",
        help="hybrid=BGE/MiniLM dense + TF-IDF sparse (default); tfidf/minilm for ablations",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Embedding device for dense models (use cuda with gpy)",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"Missing {args.candidates}", file=sys.stderr)
        return 1

    print(f"Building semantic index ({args.backend}) from {args.candidates}")
    index = SemanticIndex.build_from_candidates(
        args.candidates,
        args.out_dir,
        backend=args.backend,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Done: {len(index.id_to_row)} scores saved to {args.out_dir}")
    print(f"Backend: {index.backend}, model: {index.model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
