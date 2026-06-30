#!/usr/bin/env python3
"""Verify GTR-base encoder and vec2text corrector load on CPU."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from config_loader import load_config  # noqa: E402
from decode import decode_vectors, load_corrector  # noqa: E402
from encode import encode_texts  # noqa: E402


def verify_install(config_path: Path) -> int:
    cfg = load_config(config_path)

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
    from run_test import INPUT_CONFIG  # noqa: WPS433

    raise SystemExit(verify_install(INPUT_CONFIG))
