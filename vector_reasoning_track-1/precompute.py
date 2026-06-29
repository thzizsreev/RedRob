#!/usr/bin/env python3
"""Offline precomputation: encode 6 anchor texts → vectors/*.npy (128d)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import ANCHOR_TEXTS, DIMENSIONS
from encode import LATENT_DIM, langvae_encode

VECTORS_DIR = _ROOT / "vectors"


def main() -> None:
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    for dim in DIMENSIONS:
        v_lo_text, v_hi_text = ANCHOR_TEXTS[dim]
        np.save(VECTORS_DIR / f"v_anch_{dim}_lo.npy", langvae_encode(v_lo_text))
        np.save(VECTORS_DIR / f"v_anch_{dim}_hi.npy", langvae_encode(v_hi_text))

    sample = np.load(VECTORS_DIR / "v_anch_tech_hi.npy")
    print(
        f"Precomputation complete. 6 anchor vectors saved to vectors/ "
        f"(dim={LATENT_DIM}, shape={sample.shape})"
    )


if __name__ == "__main__":
    main()
