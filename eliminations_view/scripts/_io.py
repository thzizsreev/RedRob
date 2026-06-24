"""Shared I/O helpers for runtime artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import polars as pl


def load_json_ids(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], str):
            return [str(x) for x in data]
        return [str(row["candidate_id"]) for row in data if row.get("candidate_id")]
    raise ValueError(f"Expected JSON array in {path}")


def load_ids_from_parquet(path: Path) -> list[str]:
    return pl.read_parquet(path)["candidate_id"].cast(pl.Utf8).to_list()


def load_ids(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        return load_ids_from_parquet(path)
    return load_json_ids(path)


def parquet_to_dict(path: Path, key: str = "candidate_id") -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    out: dict[str, dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        cid = str(row[key])
        cleaned: dict[str, Any] = {}
        for k, value in row.items():
            if k == key:
                continue
            if value is None:
                cleaned[k] = None
            elif hasattr(value, "item"):
                cleaned[k] = value.item()
            else:
                cleaned[k] = value
        out[cid] = cleaned
    return out


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json_dict(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
