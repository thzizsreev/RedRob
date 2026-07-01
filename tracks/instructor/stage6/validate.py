"""Validate Stage 6 submission CSV contract."""

from __future__ import annotations

from pathlib import Path

from tracks.instructor.stage5.validate import validate_submission_csv

__all__ = ["validate_submission_csv"]


def validate_stage6_csv(
    path: Path,
    *,
    expected_rows: int,
    input_candidate_ids: set[str] | None = None,
) -> None:
    validate_submission_csv(
        path,
        expected_rows=expected_rows,
        input_candidate_ids=input_candidate_ids,
    )
