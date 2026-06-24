"""Stage 0 — offline precompute (vectors, cross-encoder export, clustering)."""

from tracks.instructor.stage0.cluster_precompute import run_cluster_precompute
from tracks.instructor.stage0.cross_encoder_export import (
    export_cross_encoder,
    load_model_id_from_config,
    run_cross_encoder_export,
    smoke_test,
)
from tracks.instructor.stage0.precompute import load_candidates_json, run_precompute

__all__ = [
    "export_cross_encoder",
    "load_candidates_json",
    "load_model_id_from_config",
    "run_cluster_precompute",
    "run_cross_encoder_export",
    "run_precompute",
    "smoke_test",
]
