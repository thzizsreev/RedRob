#!/usr/bin/env python3
"""Verify GTR-base encoder and vec2text corrector load on CPU."""

from __future__ import annotations

import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from config_loader import load_config  # noqa: E402
from decode import decode_vectors, load_corrector  # noqa: E402
from encode import encode_texts  # noqa: E402
from paths import DEFAULT_CONFIG  # noqa: E402


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)

    print(f"Loading encoder: {cfg.encoder.model_name}")
    vectors = encode_texts(cfg.encoder.model_name, ["hello world"])
    print(f"Encoder output shape: {vectors.shape}")
    if vectors.shape != (1, 768):
        print(f"ERROR: expected shape (1, 768), got {vectors.shape}")
        return 1

    print(f"Loading vec2text corrector: {cfg.decoder.embedder_id}")
    load_corrector(cfg.decoder.embedder_id)
    print("Vec2text corrector loaded successfully.")

    print("Running single decode smoke test...")
    decoded = decode_vectors(
        vectors,
        embedder_id=cfg.decoder.embedder_id,
        num_steps=cfg.decoder.num_steps,
        sequence_beam_width=cfg.decoder.sequence_beam_width,
    )[0]
    print(f"Decoded: {decoded!r}")
    print("\nInstallation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
