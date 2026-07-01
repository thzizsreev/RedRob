#!/usr/bin/env python3
"""Single-pass encode → decode round trip.

  env\\Scripts\\python.exe experiments\\vectorSteering\\feasibility\\run_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# --- paths (edit here) ---
INPUT_FILE = PACKAGE_DIR / "input.txt"
OUTPUT_FILE = PACKAGE_DIR / "results.json"

# --- model settings (edit here) ---
ENCODER_MODEL = "sentence-transformers/gtr-t5-base"
DECODER_EMBEDDER_ID = "gtr-base"


def read_input_texts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def main() -> int:
    from decode import decode_vector
    from encode import encode_text

    if not INPUT_FILE.is_file():
        print(f"ERROR: input file not found: {INPUT_FILE}")
        return 1

    texts = read_input_texts(INPUT_FILE)
    if not texts:
        print(f"ERROR: no text in {INPUT_FILE}")
        return 1

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Encoder: {ENCODER_MODEL}")
    print(f"Decoder: vec2text {DECODER_EMBEDDER_ID} (1 pass)\n")

    results = []
    for i, text in enumerate(texts, start=1):
        print(f"[{i}/{len(texts)}] INPUT: {text!r}")
        vector = encode_text(ENCODER_MODEL, text)
        print(f"         vector shape: {vector.shape}")
        decoded = decode_vector(vector, embedder_id=DECODER_EMBEDDER_ID)
        print(f"         DECODED: {decoded!r}\n")
        results.append({"input": text, "decoded": decoded, "vector_dim": int(vector.shape[0])})

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
