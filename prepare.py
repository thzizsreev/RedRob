#!/usr/bin/env python3
"""Convert sample JSON array to JSONL for pipeline testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON array to JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample_candidates.json"),
        help="Input JSON array file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample_candidates.jsonl"),
        help="Output JSONL file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print("Error: expected JSON array", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
