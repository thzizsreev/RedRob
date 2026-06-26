"""INSTRUCTOR Track A constants."""

import os
from pathlib import Path

from tracks.shared.paths import ROOT_DIR

INSTRUCTOR_MODEL = "hkunlp/instructor-large"
BLOCK_DIM = 768
VECTOR_DIM = BLOCK_DIM * 3  # 2304-d concatenated candidate vector

MAX_PASSAGE_TOKENS = 480
EMPTY_BLOCK_TEXT = "no relevant experience"

QUERY_WEIGHTS = (0.5, 0.3, 0.2)  # retrieval, infra, eval

RETRIEVAL_INSTRUCTION = (
    "Represent the AI engineering career history for retrieving candidates "
    "who have built or shipped semantic search, embeddings-based retrieval, "
    "hybrid (dense plus keyword) search, or RAG pipelines deployed to real users, "
    "including handling embedding drift, index refresh, or retrieval-quality regression:"
)

INFRA_INSTRUCTION = (
    "Represent the AI engineering career history for retrieving candidates "
    "who have operated vector databases or hybrid search infrastructure in production "
    "such as FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, or Elasticsearch, "
    "including scaling, latency optimization, or inference throughput work:"
)

EVAL_INSTRUCTION = (
    "Represent the AI engineering career history for retrieving candidates "
    "who have designed offline evaluation frameworks for ranking or retrieval systems, "
    "measured NDCG, MRR, or MAP, run online A/B tests, or built offline-to-online "
    "feedback loops to validate system improvements:"
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
CANDIDATE_VECTORS_FILENAME = "candidate_vectors.npy"
CANDIDATE_FEATURES_FILENAME = "candidate_features.parquet"
STAGE3_QUERY_VECTORS_DIR = "stage3_query_vectors"
STAGE3_QUERY_MANIFEST_FILENAME = "stage3_query_manifest.json"

INSTRUCTOR_ONNX_DIR = ROOT_DIR / "onnx" / "models"
INSTRUCTOR_ONNX_ENCODER = INSTRUCTOR_ONNX_DIR / "instructor-large-encoder.onnx"
INSTRUCTOR_ONNX_TOKENIZER = INSTRUCTOR_ONNX_DIR / "tokenizer"
INSTRUCTOR_ONNX_DENSE = INSTRUCTOR_ONNX_DIR / "dense_weight.npy"
INSTRUCTOR_ONNX_CONFIG = INSTRUCTOR_ONNX_DIR / "config.txt"
INSTRUCTOR_ONNX_MAX_SEQ_LENGTH = 512
ONNX_BATCH_SIZE = 32
ONNX_BATCH_SIZE_FALLBACK = (16, 8, 4)
CUDA_PROVIDER = "CUDAExecutionProvider"

INDEX_BATCH_SIZE = 500
PASSAGE_PREP_WORKERS: int | None = None  # None = min(8, cpu_count - 1)

STAGE1_FLOOR = 100
STAGE1_RANDOM_SEED = 42
UMAP_CLUSTERING_DIMS = 12
UMAP_N_NEIGHBORS = 20

STAGE1_DIRNAME = "stage1"
STAGE1_CANDIDATE_VECTORS_FILENAME = "candidate_vectors.npy"
STAGE1_CLUSTER_LABELS_FILENAME = "cluster_labels.npy"
STAGE1_UMAP_REDUCED_FILENAME = "umap_reduced_12d.npy"
STAGE1_CLUSTER_MANIFEST_FILENAME = "cluster_manifest.json"

STAGE1_UMAP_N_JOBS = 1  # -1 for parallel UMAP (non-reproducible; omits random_state)
STAGE1_HDBSCAN_CORE_DIST_N_JOBS = -1


def resolve_passage_prep_workers() -> int:
    if PASSAGE_PREP_WORKERS is not None:
        return max(1, PASSAGE_PREP_WORKERS)
    return max(1, min(8, (os.cpu_count() or 4) - 1))
