"""Cross-track path constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DEFAULT_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl.gz"
CANDIDATES_JSONL_PATH = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"
SAMPLE1K_PATH = DATA_DIR / "sample1k.json"
SAMPLE2_PATH = DATA_DIR / "sample2.json"
SAMPLE5K_PATH = DATA_DIR / "sample5k.json"
SAMPLE10K_PATH = DATA_DIR / "sample10k.json"
SAMPLE20K_PATH = DATA_DIR / "sample20k.json"
