"""Paths and constants for the naive resume vector DB."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

SAMPLE_CANDIDATES_PATH = PROJECT_ROOT / "data" / "sample_candidates.json"
ARTIFACTS_DIR = ROOT / "artifacts"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

INDEX_FILENAME = "resume_index.faiss"
ID_MAP_FILENAME = "id_map.json"
PASSAGES_FILENAME = "passages.jsonl"
RANK_RESULTS_FILENAME = "rank_results.json"

JD_QUERY_TEXT = (
    "Senior AI Engineer founding team role requiring production experience with "
    "embeddings-based retrieval systems, vector databases, hybrid search, "
    "end-to-end ranking systems shipped to real users at scale, and "
    "evaluation frameworks for ranking systems including NDCG MRR MAP and A/B testing"
)
