"""Cross-track path constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Instructor pipeline runtime outputs (stage 0 → 3)
RUNTIME_DIR = ARTIFACTS_DIR / "runtime"
RUNTIME_STAGE0_DIR = RUNTIME_DIR / "stage0"
RUNTIME_STAGE1_DIR = RUNTIME_DIR / "stage1"
RUNTIME_STAGE2_DIR = RUNTIME_DIR / "stage2"
RUNTIME_STAGE3_DIR = RUNTIME_DIR / "stage3"
RUNTIME_STAGE4_DIR = RUNTIME_DIR / "stage4"
RUNTIME_STAGE5_DIR = RUNTIME_DIR / "stage5"

CROSS_ENCODER_DIR = ROOT_DIR / "models" / "cross_encoder"

DEFAULT_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl.gz"
CANDIDATES_JSONL_PATH = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"
SAMPLE1K_PATH = DATA_DIR / "sample1k.json"
SAMPLE2_PATH = DATA_DIR / "sample2.json"
SAMPLE5K_PATH = DATA_DIR / "sample5k.json"
SAMPLE10K_PATH = DATA_DIR / "sample10k.json"
SAMPLE20K_PATH = DATA_DIR / "sample20k.json"
