"""Shared constants for the block-weighted vector encoding pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ANCHORS_DIR = ARTIFACTS_DIR / "anchors"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BLOCK_DIM = 384  # native BGE-small output dimension
VECTOR_DIM = BLOCK_DIM * 3  # 1152-d concatenated candidate vector

THRESHOLDS = {
    "retrieval": 0.35,
    "infra": 0.38,
    "eval": 0.32,
}

MAX_TOKENS_PER_BLOCK = 380
EMPTY_BLOCK_TEXT = "no relevant experience"

QUERY_WEIGHTS = (0.3, 0.6, 0.1)  # retrieval, infra, eval

JD_RETRIEVAL_SENTENCES = [
    "production experience with embeddings-based retrieval systems deployed to real users",
    "handling embedding drift index refresh retrieval quality regression in production",
    "sentence-transformers OpenAI embeddings BGE E5 or similar deployed to real users",
    "hybrid retrieval dense sparse production search architecture at scale",
    "vector databases Pinecone Weaviate Qdrant Milvus FAISS operational production experience",
]

JD_INFRA_SENTENCES = [
    "production experience with vector databases or hybrid search infrastructure operational",
    "deploying machine learning systems to real users at meaningful scale",
    "large-scale inference optimization distributed systems production deployment",
    "shipped end-to-end ranking search recommendation system to real users at scale",
    "production deployment latency quality tradeoffs system reliability",
]

JD_EVAL_SENTENCES = [
    "evaluation frameworks for ranking systems NDCG MRR MAP offline to online correlation",
    "A/B test interpretation recruiter feedback loops rigorous ranking evaluation",
    "offline benchmarks online A/B testing feedback loops iterative improvement",
    "how to evaluate a ranking system rigorously statistical validation",
    "learning to rank models evaluation measurement methodology",
]

QUERY_RETRIEVAL_TEXT = (
    "query: production embeddings retrieval semantic search RAG "
    "vector database experience shipped to real users at scale"
)

QUERY_INFRA_TEXT = (
    "query: deploying scaling vector database production latency "
    "optimization inference throughput operational experience"
)

QUERY_EVAL_TEXT = (
    "query: NDCG MRR MAP evaluation framework ranking A/B testing "
    "offline benchmark online experiment design feedback loop"
)

FAISS_HNSW_M = 32
FAISS_EF_CONSTRUCTION = 200
FAISS_EF_SEARCH = 64
INDEX_BATCH_SIZE = 500
SENTENCE_ENCODE_BATCH_SIZE = 64

DEFAULT_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl.gz"
