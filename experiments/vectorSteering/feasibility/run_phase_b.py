#!/usr/bin/env python3
"""Phase B: steering direction sanity check."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from acceptance import evaluate_phase_a_determinism, group_deterministic  # noqa: E402
from config_loader import load_config  # noqa: E402
from decode import decode_vectors  # noqa: E402
from encode import encode_texts  # noqa: E402
from report import PHASE_B_FIELDS, write_results_csv  # noqa: E402
from steer import compute_steering_direction, steer_vector  # noqa: E402


def phase_a_passed(output_dir: Path, phase_a_csv_name: str) -> tuple[bool, str | None]:
    csv_path = output_dir / phase_a_csv_name
    if not csv_path.exists():
        return False, f"Missing {csv_path}; run Phase A first."

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    passed, _, summary = evaluate_phase_a_determinism(rows)
    if not passed:
        return False, summary["detail"]
    return True, None


def run_phase_b(
    config_path: Path,
    output_dir: Path,
    phase_b_csv_name: str,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = cfg.phase_b.anchors
    anchor_texts = [anchors["bad"], anchors["good"], anchors["base"]]
    print(f"Encoding anchors + base ({len(anchor_texts)} sentences)...")
    anchor_vecs = encode_texts(cfg.encoder.model_name, anchor_texts)
    v_bad, v_good, v_base = anchor_vecs[0], anchor_vecs[1], anchor_vecs[2]
    v_steer = compute_steering_direction(v_good, v_bad)

    print(f"Loading decoder: {cfg.decoder.embedder_id}")
    rows: list[dict[str, Any]] = []
    base_text = anchors["base"]

    for s_value in cfg.phase_b.s_values:
        v_target = steer_vector(v_base, v_steer, s_value)
        sweep_decodes: list[str] = []

        for sweep in range(1, cfg.phase_b.sweep_repeats + 1):
            print(f"  S={s_value:+.2f} sweep {sweep}/{cfg.phase_b.sweep_repeats}...", flush=True)
            decoded = decode_vectors(
                v_target,
                embedder_id=cfg.decoder.embedder_id,
                num_steps=cfg.decoder.num_steps,
                sequence_beam_width=cfg.decoder.sequence_beam_width,
            )[0]
            sweep_decodes.append(decoded)
            rows.append(
                {
                    "phase": "B",
                    "input_id": f"steer_S{s_value:+.2f}",
                    "s_value": s_value,
                    "run_number": sweep,
                    "original_text": base_text,
                    "decoded_text": decoded,
                    "deterministic_auto": "",
                    "human_fidelity": "",
                    "human_coherent": "",
                    "human_sentiment": "",
                }
            )

        stable = group_deterministic(sweep_decodes)
        for row in rows:
            if row["s_value"] == s_value:
                row["deterministic_auto"] = "stable" if stable else "unstable"

    csv_path = output_dir / phase_b_csv_name
    write_results_csv(csv_path, rows, fieldnames=PHASE_B_FIELDS)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")
    return rows


if __name__ == "__main__":
    from run_test import (  # noqa: WPS433
        FORCE_PHASE_B,
        INPUT_CONFIG,
        OUTPUT_DIR,
        PHASE_A_CSV,
        PHASE_B_CSV,
    )

    ok, reason = phase_a_passed(OUTPUT_DIR, PHASE_A_CSV)
    if not ok and not FORCE_PHASE_B:
        print(f"Phase B skipped: {reason}")
        print("Set FORCE_PHASE_B = True in run_test.py to override.")
        raise SystemExit(1)
    run_phase_b(INPUT_CONFIG, OUTPUT_DIR, PHASE_B_CSV)
