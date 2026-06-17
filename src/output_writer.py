"""Write ranked submission CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RankedCandidate:
    candidate_id: str
    score: float
    reasoning: str
    rank: int = 0


def write_submission_csv(
    ranked: list[RankedCandidate],
    output_path: Path,
    *,
    top_n: int = 100,
) -> None:
    sorted_rows = sorted(
        ranked,
        key=lambda r: (-r.score, r.candidate_id),
    )[:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow(
                [
                    row.candidate_id,
                    rank,
                    f"{row.score:.4f}",
                    row.reasoning,
                ]
            )
