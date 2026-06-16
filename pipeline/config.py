"""Shared constants for the INSTRUCTOR block-weighted vector encoding pipeline."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

INSTRUCTOR_MODEL = "hkunlp/instructor-large"
BLOCK_DIM = 768
VECTOR_DIM = BLOCK_DIM * 3  # 2304-d concatenated candidate vector

MAX_PASSAGE_TOKENS = 480
EMPTY_BLOCK_TEXT = "no relevant experience"

QUERY_WEIGHTS = (0.5, 0.3, 0.2)  # retrieval, infra, eval

RETRIEVAL_INSTRUCTION = (
    "Represent the AI engineering career history for retrieving candidates "
    "with production experience in semantic search, embeddings-based retrieval, "
    "hybrid search systems, and ranking pipelines:"
)

INFRA_INSTRUCTION = (
    "Represent the AI engineering career history for retrieving candidates "
    "with production experience deploying and scaling ML systems, vector databases, "
    "inference optimization, and MLOps infrastructure:"
)

EVAL_INSTRUCTION = (
    "Represent the AI engineering career history for retrieving candidates "
    "who have designed evaluation frameworks for ranking systems, run A/B tests, "
    "measured NDCG, MRR, MAP, and built offline-to-online feedback loops:"
)

INSTRUCTIONS = {
    "retrieval": RETRIEVAL_INSTRUCTION,
    "infra": INFRA_INSTRUCTION,
    "eval": EVAL_INSTRUCTION,
}

JD_RETRIEVAL_TEXT = (
    "Production experience with embeddings-based retrieval systems deployed to real users. "
    "Handling embedding drift, index refresh, retrieval-quality regression in production. "
    "Production experience with vector databases or hybrid search infrastructure. "
    "Shipped at least one end-to-end ranking or search or recommendation system."
)

JD_INFRA_TEXT = (
    "Vector database scaling, FAISS, latency optimization, throughput, production deployment. "
    "Background in distributed systems or large-scale inference optimization. "
    "Deploy and maintain machine learning systems on cloud infrastructure."
)

JD_EVAL_TEXT = (
    "Hands-on experience designing evaluation frameworks for ranking systems. "
    "NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. "
    "Evaluation infrastructure, offline benchmarks, online A/B testing, recruiter feedback loops."
)

JD_QUERY_VEC_FILENAME = "jd_query_vec.npy"
INDEX_FILENAME = "candidate_index.faiss"
ID_MAP_FILENAME = "id_map.json"

INSTRUCTOR_BATCH_SIZE = 32
INSTRUCTOR_BATCH_SIZE_CPU = 8
INSTRUCTOR_VRAM_GB_ESTIMATE = 3.2
ENCODE_DEVICE = "auto"  # "auto" | "cuda" | "cpu" — precompute only

INDEX_BATCH_SIZE = 500
PASSAGE_PREP_WORKERS: int | None = None  # None = min(8, cpu_count - 1)

DEFAULT_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl.gz"
CANDIDATES_JSONL_PATH = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"


def resolve_passage_prep_workers() -> int:
    if PASSAGE_PREP_WORKERS is not None:
        return max(1, PASSAGE_PREP_WORKERS)
    return max(1, min(8, (os.cpu_count() or 4) - 1))
