#!/usr/bin/env python3
"""Generate docker/data/pool1k.jsonl via reservoir sampling from the full candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from docker.paths import DATA_DIR, MANIFEST_PATH, POOL1K_JSONL, ROOT_DIR
from tracks.instructor.core.io import iter_candidates_from_path

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def reservoir_sample(path: Path, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    n_seen = 0
    for record in iter_candidates_from_path(path):
        n_seen += 1
        cid = record.get("candidate_id")
        if not cid or not CANDIDATE_ID_PATTERN.match(str(cid)):
            continue
        if len(reservoir) < k:
            reservoir.append(record)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < k:
                reservoir[j] = record
    if len(reservoir) < k:
        raise ValueError(
            f"Only found {len(reservoir)} valid candidates in {path}; need {k}"
        )
    return reservoir


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample pool1k.jsonl for docker sandbox.")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT_DIR / "data" / "candidates.jsonl",
        help="Full candidate pool JSONL path",
    )
    parser.add_argument("--n", type=int, default=1000, help="Pool size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = reservoir_sample(args.source, args.n, args.seed)
    ids = [str(r["candidate_id"]) for r in records]

    with open(POOL1K_JSONL, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    id_hash = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    manifest = {
        "version": 1,
        "seed": args.seed,
        "n_pool": len(records),
        "candidate_ids": ids,
        "candidate_ids_hash": id_hash,
        "source_path": str(args.source.resolve()),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(records)} records to {POOL1K_JSONL}")
    print(f"Wrote manifest to {MANIFEST_PATH} (hash={id_hash})")


if __name__ == "__main__":
    main()
