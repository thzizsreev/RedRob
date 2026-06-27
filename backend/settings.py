"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from tracks.shared.paths import ROOT_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDROB_", extra="ignore")

    api_pools_root: Path = ROOT_DIR / "artifacts" / "api" / "pools"
    default_config_path: Path = ROOT_DIR / "config.yaml"
    max_workers: int = 1
    sync_jobs: bool = False
    candidate_id_pattern: str = r"^CAND_[0-9]{7}$"


@lru_cache
def get_settings() -> Settings:
    return Settings()
