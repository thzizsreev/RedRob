"""Global skill document-frequency and IDF table over the full candidate corpus."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from tracks.instructor.stage0.tier_relevance import normalize_skill_name


def _skill_names_from_record(record: dict) -> set[str]:
    names: set[str] = set()
    for skill in record.get("skills") or []:
        name = normalize_skill_name(str(skill.get("name", "")))
        if name:
            names.add(name)
    return names


def build_skill_idf(
    records: list[dict],
    *,
    persist_path: Path | None = None,
) -> tuple[dict[str, float], int]:
    """
    Compute rarity(s) = log(N / (1 + df(s))) for each skill name.

    Returns (idf_by_skill, corpus_size).
    """
    n = len(records)
    if n == 0:
        return {}, 0

    df_counter: Counter[str] = Counter()
    for record in records:
        df_counter.update(_skill_names_from_record(record))

    idf_table = {
        skill: math.log(n / (1.0 + df))
        for skill, df in df_counter.items()
    }

    if persist_path is not None:
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus_size": n,
            "skill_count": len(idf_table),
            "idf": idf_table,
        }
        with open(persist_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"Wrote skill IDF table: {persist_path} ({len(idf_table):,} skills)")

    print(f"Built skill IDF over {n:,} candidates ({len(idf_table):,} unique skills)")
    return idf_table, n
