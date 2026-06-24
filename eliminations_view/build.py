#!/usr/bin/env python3
"""
Build elimination JSON + interactive HTML for stages 2–5.

    python eliminations_view/build.py
    python eliminations_view/build.py --stage 2
    python eliminations_view/build.py --honeypots-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.shared.paths import (  # noqa: E402
    CANDIDATES_JSONL_PATH,
    ROOT_DIR,
    RUNTIME_DIR,
)

from eliminations_view.scripts.collect import collect_eliminations, write_outputs  # noqa: E402
from eliminations_view.scripts.render_html import write_html_files  # noqa: E402

DEFAULT_OUT_DIR = ROOT_DIR / "eliminations_view" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build eliminations JSON + HTML viewer for stages 2–5."
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=RUNTIME_DIR,
        help="Pipeline runtime directory (contains stage2/ … stage5/)",
    )
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_JSONL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--stage",
        type=int,
        choices=[2, 3, 4, 5],
        action="append",
        help="Build only selected stage(s); repeatable",
    )
    parser.add_argument(
        "--honeypots-only",
        action="store_true",
        help="Stage 2 only: include honeypot removals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = set(args.stage) if args.stage else None
    if args.honeypots_only and stages and 2 not in stages:
        raise ValueError("--honeypots-only requires stage 2 in --stage filter")

    out_dir = args.out.resolve()
    dataset = collect_eliminations(
        args.runtime.resolve(),
        args.candidates.resolve(),
        stages=stages,
        honeypots_only=args.honeypots_only,
    )

    json_paths = write_outputs(dataset, out_dir)
    html_paths = write_html_files(dataset, out_dir)

    manifest = {
        "built_at_utc": dataset["meta"]["built_at_utc"],
        "elimination_count": dataset["meta"]["elimination_count"],
        "counts_by_stage": dataset["meta"]["counts_by_stage"],
        "reason_breakdown": dataset["meta"]["reason_breakdown"],
        "sources": dataset["meta"]["sources"],
        "outputs": {**json_paths, **html_paths},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Eliminations: {dataset['meta']['elimination_count']}")
    for stage, count in sorted(dataset["meta"]["counts_by_stage"].items()):
        print(f"  Stage {stage}: {count}")
    if dataset["meta"]["missing_profiles"]:
        print(
            f"  Warning: {dataset['meta']['missing_profiles']} profile(s) missing from JSONL"
        )
    print(f"Wrote {out_dir / 'all_eliminations.html'}")


if __name__ == "__main__":
    main()
