"""Configuration for the honeypot research pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from tracks.shared.paths import ROOT_DIR, SAMPLE1K_PATH

HONEYPOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HONEYPOT_DIR / "output"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_WORKERS = 4
DEFAULT_REQUESTS_PER_MINUTE = 60
DEFAULT_MAX_RETRIES = 6
DEFAULT_INITIAL_BACKOFF_SEC = 2.0
DEFAULT_PER_STRATUM = 50
DEFAULT_RANDOM_SEED = 42

PASS1_RESULTS_FILENAME = "pass1_results.jsonl"
PASS2_RESULTS_FILENAME = "pass2_results.jsonl"
FAILURES_FILENAME = "failures.jsonl"
MANIFEST_FILENAME = "manifest.json"
RUN_SUMMARY_FILENAME = "run_summary.json"


@dataclass(frozen=True)
class PipelineConfig:
    candidates_path: Path
    output_dir: Path
    openai_model: str
    max_workers: int
    requests_per_minute: int
    max_retries: int
    initial_backoff_sec: float
    per_stratum: int
    random_seed: int
    force: bool = False
    sample_only: bool = False
    pass_mode: str = "all"  # "1", "2", or "all"
    manifest_path: Path | None = None
    filtered_ids_path: Path | None = None
    verbose: bool = False


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env")


def get_openai_api_key() -> str:
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def resolve_openai_model() -> str:
    load_env()
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


def default_config(**overrides) -> PipelineConfig:
    """Build config from overrides; path defaults are set in honeypot/run.py."""
    load_env()
    base = {
        "candidates_path": SAMPLE1K_PATH,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "filtered_ids_path": None,
        "openai_model": resolve_openai_model(),
        "max_workers": DEFAULT_MAX_WORKERS,
        "requests_per_minute": DEFAULT_REQUESTS_PER_MINUTE,
        "max_retries": DEFAULT_MAX_RETRIES,
        "initial_backoff_sec": DEFAULT_INITIAL_BACKOFF_SEC,
        "per_stratum": DEFAULT_PER_STRATUM,
        "random_seed": DEFAULT_RANDOM_SEED,
    }
    base.update(overrides)
    return PipelineConfig(**base)
