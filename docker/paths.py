"""Path constants for the docker sandbox."""

from pathlib import Path

DOCKER_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCKER_DIR.parent

DATA_DIR = DOCKER_DIR / "data"
WORK_DIR = DOCKER_DIR / "work"
POOL1K_JSONL = DATA_DIR / "pool1k.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

PRECOMPUTED_POOL1K = DOCKER_DIR / "artifacts" / "precomputed" / "pool1k"
POOL1K_STAGE0 = PRECOMPUTED_POOL1K / "stage0"
POOL1K_STAGE1 = PRECOMPUTED_POOL1K / "stage1"

WORK_RUNTIME = WORK_DIR / "runtime"
WORK_DATA = WORK_DIR / "data"
WORK_CONFIG = WORK_DIR / "config.runtime.yaml"
ACTIVE_JSONL = WORK_DATA / "active.jsonl"

SANDBOX_CONFIG = DOCKER_DIR / "config.yaml"
DEFAULT_OUTPUT_CONTAINER = Path("/output")
