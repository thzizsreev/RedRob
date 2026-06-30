#!/usr/bin/env python3
"""Phase A: baseline encode-decode stability (no steering)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from acceptance import group_deterministic  # noqa: E402
from config_loader import load_config  # noqa: E402
from decode import decode_vectors  # noqa: E402
from encode import encode_texts  # noqa: E402
from report import PHASE_A_FIELDS, write_results_csv  # noqa: E402


def run_phase_a(
    config_path: Path,
    output_dir: Path,
    phase_a_csv_name: str,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [s.text for s in cfg.phase_a.sentences]
    ids = [s.id for s in cfg.phase_a.sentences]

    print(f"Loading encoder: {cfg.encoder.model_name}")
    vectors = encode_texts(cfg.encoder.model_name, texts)
    print(f"Encoded {len(texts)} sentences -> shape {vectors.shape}")

    print(f"Loading decoder: {cfg.decoder.embedder_id}")
    rows: list[dict[str, Any]] = []

    for idx, (sentence_id, original_text) in enumerate(zip(ids, texts, strict=True)):
        vector = vectors[idx]
        decoded_runs: list[str] = []

        for run_num in range(1, cfg.phase_a.decode_repeats + 1):
            print(f"  {sentence_id} decode run {run_num}/{cfg.phase_a.decode_repeats}...", flush=True)
            decoded = decode_vectors(
                vector,
                embedder_id=cfg.decoder.embedder_id,
                num_steps=cfg.decoder.num_steps,
                sequence_beam_width=cfg.decoder.sequence_beam_width,
            )[0]
            decoded_runs.append(decoded)
            rows.append(
                {
                    "phase": "A",
                    "input_id": sentence_id,
                    "s_value": "",
                    "run_number": run_num,
                    "original_text": original_text,
                    "decoded_text": decoded,
                    "deterministic_auto": "",
                    "human_fidelity": "",
                    "human_coherent": "",
                    "human_sentiment": "",
                }
            )

        stable = group_deterministic(decoded_runs)
        for row in rows:
            if row["input_id"] == sentence_id:
                row["deterministic_auto"] = "stable" if stable else "unstable"

    csv_path = output_dir / phase_a_csv_name
    write_results_csv(csv_path, rows, fieldnames=PHASE_A_FIELDS)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")
    return rows


if __name__ == "__main__":
    from run_test import INPUT_CONFIG, OUTPUT_DIR, PHASE_A_CSV  # noqa: WPS433

    run_phase_a(INPUT_CONFIG, OUTPUT_DIR, PHASE_A_CSV)
