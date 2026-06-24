"""
INSTRUCTOR track shared primitives — config, ONNX embedder, encoding, passage extraction,
FAISS index build, and artifact I/O.

Used by Stage 0 precompute and runtime stages (1–5). Stage runners live under stageN/.
"""

from tracks.instructor.core.config import (
    INSTRUCTOR_MODEL,
    QUERY_WEIGHTS,
    VECTOR_DIM,
)
from tracks.instructor.core.encode import (
    build_jd_query_vector,
    encode_candidates,
    load_tokenizer,
    log_encode_plan,
)
from tracks.instructor.core.extraction import build_candidate_passage, build_candidate_segments
from tracks.instructor.core.index import build_vector_index, build_vector_index_from_records
from tracks.instructor.core.io import (
    iter_candidates_from_path,
    load_candidate_ids_from_id_map,
    load_index_and_id_map,
    load_jd_query_vector,
    load_vectors_from_artifacts,
)
from tracks.instructor.core.onnx_embedder import InstructorONNX, load_embedder, unload_embedder

__all__ = [
    "INSTRUCTOR_MODEL",
    "InstructorONNX",
    "QUERY_WEIGHTS",
    "VECTOR_DIM",
    "build_candidate_passage",
    "build_candidate_segments",
    "build_jd_query_vector",
    "build_vector_index",
    "build_vector_index_from_records",
    "encode_candidates",
    "iter_candidates_from_path",
    "load_candidate_ids_from_id_map",
    "load_embedder",
    "load_index_and_id_map",
    "load_jd_query_vector",
    "load_tokenizer",
    "load_vectors_from_artifacts",
    "log_encode_plan",
    "unload_embedder",
]
