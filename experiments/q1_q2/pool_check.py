#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))

from encode import build_q1_vector, build_q2_vector, load_vector_config
from paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_CONFIG, validate_artifacts
from score import ranks_from_scores, score_all_candidates, score_at_id

config = load_vector_config(DEFAULT_CONFIG)
cands = json.loads((TEST_DIR / "input/test_candidates.json").read_text(encoding="utf-8"))
rows = []
for tier in ("tier_A_keyword_rich", "tier_B_outcome_language", "tier_C_weak_tail", "tier_D_phd_research"):
    for entry in cands[tier]:
        rows.append((entry["id"], entry["name"], tier))

from experiments.stage3.shared.cpu_embedder import load_embedder

model = load_embedder()
q1 = build_q1_vector(model, config)
q2 = build_q2_vector(model, config)
mat, all_ids = validate_artifacts(
    DEFAULT_ARTIFACTS_DIR / "candidate_vectors.npy",
    DEFAULT_ARTIFACTS_DIR / "id_map.json",
)
q1s = score_all_candidates(q1, mat)
q1r = ranks_from_scores(q1s)
q2s = score_all_candidates(q2, mat)
q2r = ranks_from_scores(q2s)
print(f"Pool size: {len(all_ids):,}")
print(f"{'ID':<14} {'name':<20} {'tier':<28} {'Q1 rank':>8} {'Q2 rank':>8}")
for cid, name, tier in rows:
    _, r1 = score_at_id(q1s, q1r, all_ids, cid)
    _, r2 = score_at_id(q2s, q2r, all_ids, cid)
    print(f"{cid:<14} {name[:20]:<20} {tier:<28} {r1:>8} {r2:>8}")
