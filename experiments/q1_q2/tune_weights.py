#!/usr/bin/env python3
"""Search facet combination weights (pre-encoded vectors, fast grid)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))

from acceptance import evaluate_config_pass  # noqa: E402
from encode import encode_facet, load_vector_config  # noqa: E402
from paths import DEFAULT_CONFIG, DEFAULT_SYNTHETIC  # noqa: E402

STEP = 0.05


def _simplex_grid(n: int, step: float) -> list[tuple[float, ...]]:
    ticks = np.arange(step, 1.0, step)
    ticks = np.round(ticks, 10)
    out: list[tuple[float, ...]] = []
    for combo in itertools.product(ticks, repeat=n - 1):
        rest = 1.0 - sum(combo)
        if rest < step - 1e-9 or rest > 1.0 - step + 1e-9:
            continue
        w = tuple(round(x, 2) for x in (*combo, rest))
        if abs(sum(w) - 1.0) > 1e-6:
            continue
        out.append(w)
    return out


def _score_cases(
    cand_vecs: dict[str, np.ndarray],
    q1_matrix: np.ndarray,
    q1_w: np.ndarray,
    q2_matrix: np.ndarray,
    q2_w: np.ndarray,
) -> dict[str, dict[str, float]]:
    q1 = q1_matrix.T @ q1_w
    q2 = q2_matrix.T @ q2_w
    scores: dict[str, dict[str, float]] = {}
    for case_id, vec in cand_vecs.items():
        scores[case_id] = {
            "q1": float(np.dot(vec, q1)),
            "q2": float(np.dot(vec, q2)),
        }
    return scores


def _margin_loss(scores: dict[str, dict[str, float]]) -> float:
    """Negative sum of constraint margins (lower = better)."""
    s = scores
    margins = [
        s["TC1"]["q1"] - 0.90,
        s["TC2"]["q1"] - 0.87,
        s["TC2"]["q2"] - 0.86,
        s["TC3"]["q1"] - 0.88,
        0.78 - s["TC4"]["q2"],
        (s["TC4"]["q1"] - s["TC4"]["q2"]) - 0.06,
        0.82 - s["TC5"]["q1"],
        (s["TC1"]["q1"] - s["TC5"]["q1"]) - 0.08,
    ]
    return -sum(min(m, 0.0) for m in margins)


def main() -> int:
    config = load_vector_config(DEFAULT_CONFIG)
    with open(DEFAULT_SYNTHETIC, encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]

    from experiments.stage3.shared.cpu_embedder import load_embedder  # noqa: WPS433
    from tracks.instructor.core.encode import encode_candidates  # noqa: WPS433

    print("Encoding facet + candidate vectors (once)...")
    model = load_embedder()

    q1_ids = [f.id for f in config.q1_facets]
    q2_ids = [f.id for f in config.q2_facets]
    q1_matrix = np.stack(
        [encode_facet(model, f.text, config.q1_subspace) for f in config.q1_facets],
        axis=0,
    )
    q2_matrix = np.stack(
        [encode_facet(model, f.text, config.q2_subspace) for f in config.q2_facets],
        axis=0,
    )
    cand_vecs = {
        c["id"]: encode_candidates(model, [c["text"].strip()], batch_size=1)[0]
        for c in cases
    }

    q1_grids = _simplex_grid(4, STEP)
    q2_grids = _simplex_grid(3, STEP)
    print(f"Q1 combos: {len(q1_grids)}, Q2 combos: {len(q2_grids)}")

    best_pass: tuple | None = None
    best_loss = float("inf")
    best_fail: tuple | None = None

    for q1_w in q1_grids:
        q1_arr = np.array(q1_w, dtype=np.float32)
        for q2_w in q2_grids:
            q2_arr = np.array(q2_w, dtype=np.float32)
            scores = _score_cases(cand_vecs, q1_matrix, q1_arr, q2_matrix, q2_arr)
            passed, _ = evaluate_config_pass(case_scores=scores)
            if passed:
                best_pass = (q1_w, q2_w, scores)
                break
            loss = _margin_loss(scores)
            if loss < best_loss:
                best_loss = loss
                best_fail = (q1_w, q2_w, scores)
        if best_pass:
            break

    if best_pass:
        q1_w, q2_w, scores = best_pass
        print("FOUND PASSING WEIGHTS:")
    elif best_fail:
        q1_w, q2_w, scores = best_fail
        print("No passing combo in grid. Best margin (closest):")
    else:
        print("No results")
        return 1

    print("Q1:", dict(zip(q1_ids, q1_w, strict=True)))
    print("Q2:", dict(zip(q2_ids, q2_w, strict=True)))
    for cid in ["TC1", "TC2", "TC3", "TC4", "TC5"]:
        print(f"  {cid}: Q1={scores[cid]['q1']:.4f} Q2={scores[cid]['q2']:.4f}")
    passed, checks = evaluate_config_pass(case_scores=scores)
    print(f"PASS: {passed}")
    for chk in checks:
        print(f"  [{'ok' if chk['passed'] else 'FAIL'}] {chk['rule']}: {chk['detail']}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
