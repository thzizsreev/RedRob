"""Config hashing for precompute manifest invalidation."""

from __future__ import annotations

import hashlib
import json

from test_stage_3.shared.config_precompute import PrecomputeConfig


def query_config_hash(config: PrecomputeConfig) -> str:
    payload = {
        "q1_text": config.q1_text,
        "q2_text": config.q2_text,
        "q3_text": config.q3_text,
        "subspace_weights_q1": config.subspace_weights_q1.as_tuple(),
        "subspace_weights_q2": config.subspace_weights_q2.as_tuple(),
        "subspace_weights_q3": config.subspace_weights_q3.as_tuple(),
    }
    return _sha256(payload)


def cohort_config_hash(config: PrecomputeConfig) -> str:
    payload = {
        "stage2_source_path": str(config.stage2_source_path),
        "candidates_jsonl_path": str(config.candidates_jsonl_path),
        "sample_size": config.sample_size,
        "random_seed": config.random_seed,
        "must_have_keywords": config.must_have_keywords,
    }
    return _sha256(payload)


def _sha256(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
