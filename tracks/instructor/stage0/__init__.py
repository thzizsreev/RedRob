"""Stage 0 — offline precompute (vectors, cross-encoder export, clustering)."""

from tracks.instructor.stage0.cluster_precompute import run_cluster_precompute
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

_CROSS_ENCODER_EXPORTS = frozenset(
    {
        "export_cross_encoder",
        "load_model_id_from_config",
        "run_cross_encoder_export",
        "smoke_test",
    }
)


def __getattr__(name: str):
    if name in _CROSS_ENCODER_EXPORTS:
        from tracks.instructor.stage0 import cross_encoder_export

        return getattr(cross_encoder_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
