"""Paths and constants for the naive resume vector DB."""

from pathlib import Path

from tracks.shared.paths import ROOT_DIR, SAMPLE1K_PATH, SAMPLE_CANDIDATES_PATH

TRACK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = TRACK_DIR / "artifacts"
NAIVE_ARTIFACTS_DIR = ARTIFACTS_DIR
NAIVE_INDEX_FILENAME = "resume_index.faiss"
NAIVE_VECTOR_DIM = 384  # BGE-small-en-v1.5

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = NAIVE_VECTOR_DIM

INDEX_FILENAME = NAIVE_INDEX_FILENAME
ID_MAP_FILENAME = "id_map.json"
PASSAGES_FILENAME = "passages.jsonl"
RANK_RESULTS_FILENAME = "rank_results.json"

JD_QUERY_TEXT = (
    "Senior AI Engineer founding team role requiring production experience with "
    "embeddings-based retrieval systems, vector databases, hybrid search, "
    "end-to-end ranking systems shipped to real users at scale, and "
    "evaluation frameworks for ranking systems including NDCG MRR MAP and A/B testing"
)
