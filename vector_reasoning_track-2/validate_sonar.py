#!/usr/bin/env python3
"""Check SONAR encoder/decoder imports before running the experiment."""

from __future__ import annotations

import sys


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    try:
        from sonar.inference_pipelines.text import (  # noqa: F401
            EmbeddingToTextModelPipeline,
            TextToEmbeddingModelPipeline,
        )
    except ImportError as exc:
        print(f"FAIL: {exc}")
        print()
        print("SONAR requires fairseq2 + fairseq2n (Linux or WSL).")
        print("On Windows native Python, fairseq2n wheels are unavailable.")
        print("Use Python 3.10/3.11 on Linux or WSL - see README.md.")
        return 1

    print("PASS: sonar.inference_pipelines.text imports OK")
    print("Run: python precompute.py && python run_experiment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
