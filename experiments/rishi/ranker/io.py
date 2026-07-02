"""Streaming I/O, top-K heap, and post-rank trap filter for TRACER."""

from __future__ import annotations

import csv
import gzip
import heapq
import json
from pathlib import Path
from typing import Iterator

from ranker import jd_config as jd
from ranker.embeddings import SemanticIndex
from ranker.reasoning import build_reasoning
from ranker.scorer import score_candidate
from ranker.submission_safe import is_submission_safe


def iter_candidates(path: Path) -> Iterator[dict]:
    """Stream candidates from .jsonl, .jsonl.gz, or .json array."""
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return

    opener = gzip.open if suffixes.endswith(".jsonl.gz") or path.suffix.lower() == ".gz" else open
    mode = "rt" if opener is gzip.open else "r"
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rank_candidates(
    path: Path,
    top_k: int = 100,
    artifacts_dir: Path | None = None,
    heap_buffer: int | None = None,
) -> list[tuple[float, str, dict, dict]]:
    """
    Score all candidates, keep buffered heap, post-filter traps, return top_k.
    """
    embed_index: SemanticIndex | None = None
    artifacts_dir = artifacts_dir or (Path(__file__).resolve().parents[1] / jd.ARTIFACTS_DIR)
    meta_path = artifacts_dir / "meta.json"
    if meta_path.exists():
        embed_index = SemanticIndex.load(artifacts_dir)

    buffer = heap_buffer or jd.HEAP_BUFFER
    heap: list[tuple[float, str, dict, dict]] = []

    for candidate in iter_candidates(path):
        cid = candidate["candidate_id"]
        sem = embed_index.semantic_score(cid) if embed_index else None
        score, context = score_candidate(candidate, semantic_score=sem)
        entry = (score, cid, candidate, context)

        if len(heap) < buffer:
            heapq.heappush(heap, entry)
        elif score > heap[0][0] or (score == heap[0][0] and cid < heap[0][1]):
            heapq.heapreplace(heap, entry)

    ranked = sorted(heap, key=lambda x: (-x[0], x[1]))

    safe: list[tuple[float, str, dict, dict]] = []
    skipped = 0
    for entry in ranked:
        _, cid, candidate, context = entry
        ok, reason = is_submission_safe(candidate, context)
        if ok:
            safe.append(entry)
        else:
            skipped += 1
            if skipped <= 5:
                print(f"[tracer] post-filter skip {cid}: {reason}")

        if len(safe) >= top_k:
            break

    if len(safe) < top_k:
        raise RuntimeError(
            f"Only {len(safe)} safe candidates after post-filter (need {top_k}). "
            "Check honeypot rules or heap buffer."
        )

    return safe[:top_k]


def write_submission(
    ranked: list[tuple[float, str, dict, dict]],
    out_path: Path,
    score_start: float = 0.990,
    score_step: float = 0.008,
) -> None:
    """Write submission CSV with monotonic scores and evidence reasoning."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_idx, (_raw_score, cid, candidate, context) in enumerate(ranked, start=1):
            score = round(score_start - (rank_idx - 1) * score_step, 4)
            reasoning = build_reasoning(rank_idx, candidate, context)
            writer.writerow([cid, rank_idx, f"{score:.4f}", reasoning])
