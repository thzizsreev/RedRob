"""Validate Stage 5 submission CSV contract."""

from __future__ import annotations

import csv
import re
from pathlib import Path

_CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")
_RANKING_HEADER = ["candidate_id", "rank", "score"]
_SUBMISSION_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def _validate_rank_score_rows(
    rows: list[dict],
    expected_rows: int,
    *,
    input_candidate_ids: set[str] | None,
    require_reasoning: bool,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} data rows, got {len(rows)}")

    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, expected_rows + 1)):
        raise ValueError("Ranks must be exactly 1..N each once")

    cids = [r["candidate_id"] for r in rows]
    if len(set(cids)) != len(cids):
        raise ValueError("All candidate_ids in submission must be unique")

    scores = [float(r["score"]) for r in rows]
    for i, score in enumerate(scores):
        if score != score:
            raise ValueError(f"NaN score at rank {i + 1}")
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            raise ValueError(
                f"Scores must be monotonically non-increasing; "
                f"rank {i + 1} ({scores[i]}) < rank {i + 2} ({scores[i + 1]})"
            )

    for row in rows:
        cid = row["candidate_id"]
        if not _CANDIDATE_ID_RE.match(cid):
            raise ValueError(f"Invalid candidate_id format: {cid}")
        if require_reasoning and not str(row.get("reasoning", "")).strip():
            raise ValueError(f"Empty reasoning for {cid}")

    if input_candidate_ids is not None:
        unknown = set(cids) - input_candidate_ids
        if unknown:
            examples = sorted(unknown)[:5]
            raise ValueError(
                f"Submission contains candidate_ids not in Stage 4 input. Examples: {examples}"
            )


def validate_ranking_csv(
    path: Path,
    expected_rows: int = 100,
    *,
    input_candidate_ids: set[str] | None = None,
) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _RANKING_HEADER:
            raise ValueError(
                f"CSV header must be {','.join(_RANKING_HEADER)}; got {reader.fieldnames}"
            )
        rows = list(reader)
    _validate_rank_score_rows(
        rows,
        expected_rows,
        input_candidate_ids=input_candidate_ids,
        require_reasoning=False,
    )


def validate_submission_csv(
    path: Path,
    expected_rows: int = 100,
    *,
    input_candidate_ids: set[str] | None = None,
) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _SUBMISSION_HEADER:
            raise ValueError(
                f"CSV header must be {','.join(_SUBMISSION_HEADER)}; got {reader.fieldnames}"
            )
        rows = list(reader)

    _validate_rank_score_rows(
        rows,
        expected_rows,
        input_candidate_ids=input_candidate_ids,
        require_reasoning=True,
    )
