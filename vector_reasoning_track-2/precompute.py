#!/usr/bin/env python3
"""Offline precomputation: encode templates and anchors → vectors/*.npy (128d)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import (
    anchor_behav_hi,
    anchor_behav_lo,
    anchor_career_hi,
    anchor_career_lo,
    anchor_tech_hi,
    anchor_tech_lo,
    template_behav,
    template_career,
    template_tech,
)
from encode import LATENT_DIM, langvae_encode

VECTORS_DIR = _ROOT / "vectors"


def main() -> None:
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    for bucket_name, text in template_tech.items():
        np.save(VECTORS_DIR / f"v_tmpl_tech_{bucket_name}.npy", langvae_encode(text))
    for bucket_name, text in template_career.items():
        np.save(VECTORS_DIR / f"v_tmpl_career_{bucket_name}.npy", langvae_encode(text))
    for bucket_name, text in template_behav.items():
        np.save(VECTORS_DIR / f"v_tmpl_behav_{bucket_name}.npy", langvae_encode(text))

    np.save(VECTORS_DIR / "v_anch_tech_hi.npy", langvae_encode(anchor_tech_hi))
    np.save(VECTORS_DIR / "v_anch_tech_lo.npy", langvae_encode(anchor_tech_lo))
    np.save(VECTORS_DIR / "v_anch_career_hi.npy", langvae_encode(anchor_career_hi))
    np.save(VECTORS_DIR / "v_anch_career_lo.npy", langvae_encode(anchor_career_lo))
    np.save(VECTORS_DIR / "v_anch_behav_hi.npy", langvae_encode(anchor_behav_hi))
    np.save(VECTORS_DIR / "v_anch_behav_lo.npy", langvae_encode(anchor_behav_lo))

    sample = np.load(VECTORS_DIR / "v_tmpl_tech_high.npy")
    print(f"Precomputation complete. 15 vectors saved to vectors/ (dim={LATENT_DIM}, shape={sample.shape})")


if __name__ == "__main__":
    main()
