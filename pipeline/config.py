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

RETRIEVAL_ANCHOR_SENTENCES = [
    "Built and deployed a hybrid retrieval system combining dense embeddings and BM25 for a production search product serving millions of users.",
    "Designed semantic search pipelines using sentence transformers and approximate nearest neighbor indexes at a product company.",
    "Owned end-to-end RAG pipeline from embedding generation to retrieval to LLM response synthesis in a live product.",
    "Shipped embedding-based candidate matching system replacing keyword search, improving precision at a recruiting platform.",
    "Managed embedding model versioning and index refresh pipelines to handle semantic drift in production retrieval.",
    "Implemented cross-encoder re-ranking on top of bi-encoder retrieval to improve top-k precision in a real-world search system.",
    "Built recommendation engine using vector similarity search on user and item embeddings deployed to production.",
    "Designed retrieval layer for a conversational AI product using FAISS with real-time document ingestion.",
    "Developed dense passage retrieval system using BGE and E5 embeddings for an enterprise knowledge base.",
    "Replaced TF-IDF search with dense retrieval and saw measurable improvement in recruiter engagement metrics.",
    "Built document retrieval system handling cold-start and embedding drift for a live SaaS product.",
    "Shipped neural information retrieval system using bi-encoders, maintaining index freshness under continuous data ingestion.",
]

INFRA_ANCHOR_SENTENCES = [
    "Scaled FAISS HNSW index to tens of millions of vectors while maintaining sub-100ms P99 query latency in production.",
    "Operated Qdrant vector database in production with real-time index updates, replication, and zero-downtime deploys.",
    "Optimized embedding inference throughput using ONNX Runtime, reducing latency by 60% compared to PyTorch baseline.",
    "Designed and maintained Pinecone index serving live search traffic with automated dimension management and namespace isolation.",
    "Built distributed vector search infrastructure on Weaviate with horizontal scaling across multiple nodes.",
    "Managed Milvus cluster for a high-throughput recommendation system, handling index partitioning and memory optimization.",
    "Reduced vector search latency from 800ms to 40ms by switching from flat index to HNSW with tuned ef_search parameters.",
    "Implemented batched embedding generation pipeline with async queuing to handle spiky ingestion workloads.",
    "Deployed OpenSearch with k-NN plugin as hybrid search backend, handling both keyword and vector queries in one system.",
    "Ran capacity planning and cost optimization for a vector database serving 200K daily active users.",
    "Built monitoring and alerting for retrieval system degradation including embedding drift detection and query latency SLOs.",
    "Designed index segmentation strategy for multi-tenant vector search with strict data isolation requirements.",
]


EVAL_ANCHOR_SENTENCES = [
    "Designed offline evaluation framework using NDCG@10 and MAP to benchmark retrieval quality before production deploys.",
    "Built A/B testing infrastructure for ranking system changes and interpreted statistical significance of online experiment results.",
    "Established correlation between offline NDCG benchmarks and online recruiter engagement metrics to validate evaluation methodology.",
    "Implemented feedback loop collecting recruiter click and save signals to continuously improve ranking model quality.",
    "Designed human relevance labeling pipeline and computed inter-annotator agreement to build reliable evaluation datasets.",
    "Ran learning-to-rank experiments using XGBoost with NDCG@10 as the training objective on collected implicit feedback.",
    "Built evaluation harness for RAG system measuring retrieval recall and generation faithfulness using automated metrics.",
    "Tracked MRR and P@K metrics across model versions using a reproducible benchmark suite tied to CI/CD pipeline.",
    "Designed holdout evaluation set for candidate ranking system and measured precision degradation over time as embedding drift occurred.",
    "Interpreted A/B test results for ranking changes accounting for novelty effects and position bias in click data.",
    "Built dashboard tracking online and offline ranking metrics to detect model decay and trigger retraining.",
    "Wrote evaluation framework distinguishing between retrieval failures and re-ranking failures to isolate system improvement levers.",
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

INDEX_BATCH_SIZE = 500
SENTENCE_ENCODE_BATCH_SIZE = 64

DEFAULT_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl.gz"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"
