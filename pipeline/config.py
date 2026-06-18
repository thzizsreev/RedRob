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

INSTRUCTOR_ONNX_DIR = ROOT_DIR / "onnx" / "models"
INSTRUCTOR_ONNX_ENCODER = INSTRUCTOR_ONNX_DIR / "instructor-large-encoder.onnx"
INSTRUCTOR_ONNX_TOKENIZER = INSTRUCTOR_ONNX_DIR / "tokenizer"
INSTRUCTOR_ONNX_DENSE = INSTRUCTOR_ONNX_DIR / "dense_weight.npy"
INSTRUCTOR_ONNX_CONFIG = INSTRUCTOR_ONNX_DIR / "config.txt"
INSTRUCTOR_ONNX_MAX_SEQ_LENGTH = 512
ONNX_BATCH_SIZE = 32
ONNX_BATCH_SIZE_FALLBACK = (16, 8, 4)
CUDA_PROVIDER = "CUDAExecutionProvider"

# Legacy path (pipeline/parallel.py) — Track A precompute uses ONNX CUDA only
ENCODE_DEVICE = "auto"

INDEX_BATCH_SIZE = 500
PASSAGE_PREP_WORKERS: int | None = None  # None = min(8, cpu_count - 1)

DEFAULT_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl.gz"
CANDIDATES_JSONL_PATH = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"
SAMPLE2_PATH = DATA_DIR / "sample2.json"
SAMPLE5K_PATH = DATA_DIR / "sample5k.json"
SAMPLE10K_PATH = DATA_DIR / "sample10k.json"


def resolve_passage_prep_workers() -> int:
    if PASSAGE_PREP_WORKERS is not None:
        return max(1, PASSAGE_PREP_WORKERS)
    return max(1, min(8, (os.cpu_count() or 4) - 1))
