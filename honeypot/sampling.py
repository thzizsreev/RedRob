"""Stratified sampling for honeypot research study."""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from honeypot.config import MANIFEST_FILENAME

# --- keyword / pattern sets ---

AI_TERMS = re.compile(
    r"\b("
    r"ml|machine learning|deep learning|ai|artificial intelligence|"
    r"embedding|embeddings|retrieval|search|ranking|recommendation|"
    r"faiss|pytorch|tensorflow|nlp|rag|llm|vector|neural|"
    r"transformer|bert|gpt|inference|model training|data science"
    r")\b",
    re.IGNORECASE,
)

CANONICAL_AI_VOCAB = re.compile(
    r"\b(rag|embeddings?|pinecone|weaviate|qdrant|milvus|faiss|"
    r"vector database|hybrid search|semantic search|llm|transformer)\b",
    re.IGNORECASE,
)

PRODUCTION_SYSTEM_LANG = re.compile(
    r"\b(recommendation system|ranking system|search system|"
    r"retrieval pipeline|relevance scoring|click-through|conversion)\b",
    re.IGNORECASE,
)

NON_TECH_TITLE = re.compile(
    r"\b(marketing|operations|sales|design|hr|human resources|"
    r"account manager|brand|content|recruiter|finance|legal)\b",
    re.IGNORECASE,
)

TECH_SKILL_NAMES = re.compile(
    r"\b(python|pytorch|tensorflow|kafka|kubernetes|docker|sql|spark|"
    r"feature engineering|machine learning|deep learning|nlp|rag|"
    r"embeddings?|faiss|transformer|llm|pytorch|scikit)\b",
    re.IGNORECASE,
)

CV_SPEECH_ROBOTICS = re.compile(
    r"\b(computer vision|opencv|object detection|speech recognition|"
    r"robotics|ros\b|slam|autonomous vehicle|yolo|cnn for images)\b",
    re.IGNORECASE,
)

NLP_IR_TERMS = re.compile(
    r"\b(nlp|information retrieval|semantic search|embeddings?|"
    r"ranking|retrieval|rag|search engine|relevance)\b",
    re.IGNORECASE,
)

CONSULTING_FIRMS = frozenset(
    {
        "tcs",
        "tata consultancy services",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
    }
)

STRATUM_B_BUCKETS = (
    "keyword_stuffer",
    "tier5_plain_language",
    "behavioral_rescue",
    "consulting_only",
    "cv_speech_robotics",
)


@dataclass(frozen=True)
class SampleEntry:
    candidate_id: str
    tags: list[str]
    stratum: str


def _normalize_company(name: str) -> str:
    return name.strip().lower()


def _narrative_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    profile = record.get("profile") or {}
    parts.append(str(profile.get("summary", "")))
    parts.append(str(profile.get("headline", "")))
    for role in record.get("career_history") or []:
        parts.append(str(role.get("description", "")))
        parts.append(str(role.get("title", "")))
    for skill in record.get("skills") or []:
        parts.append(str(skill.get("name", "")))
    return " ".join(parts)


def _has_ai_signal(record: dict[str, Any]) -> bool:
    return bool(AI_TERMS.search(_narrative_text(record)))


def _has_weak_technical_signal(record: dict[str, Any]) -> bool:
    text = _narrative_text(record)
    return not bool(
        AI_TERMS.search(text)
        or CANONICAL_AI_VOCAB.search(text)
        or PRODUCTION_SYSTEM_LANG.search(text)
    )


def _is_keyword_stuffer(record: dict[str, Any]) -> bool:
    profile = record.get("profile") or {}
    title = str(profile.get("current_title", ""))
    headline = str(profile.get("headline", ""))
    if not (NON_TECH_TITLE.search(title) or NON_TECH_TITLE.search(headline)):
        return False
    for skill in record.get("skills") or []:
        name = str(skill.get("name", ""))
        if TECH_SKILL_NAMES.search(name):
            return True
    return False


def _is_tier5_plain_language(record: dict[str, Any]) -> bool:
    text = _narrative_text(record)
    return bool(PRODUCTION_SYSTEM_LANG.search(text)) and not bool(
        CANONICAL_AI_VOCAB.search(text)
    )


def _is_behavioral_rescue(record: dict[str, Any]) -> bool:
    signals = record.get("redrob_signals") or {}
    response_rate = float(signals.get("recruiter_response_rate") or 0)
    views = int(signals.get("profile_views_received_30d") or 0)
    strong_behavior = response_rate >= 0.5 or views >= 50
    return strong_behavior and _has_weak_technical_signal(record)


def _is_consulting_only(record: dict[str, Any]) -> bool:
    history = record.get("career_history") or []
    if not history:
        return False
    for role in history:
        company = _normalize_company(str(role.get("company", "")))
        if not any(firm in company for firm in CONSULTING_FIRMS):
            return False
    return True


def _is_cv_speech_robotics(record: dict[str, Any]) -> bool:
    text = _narrative_text(record)
    return bool(CV_SPEECH_ROBOTICS.search(text)) and not bool(NLP_IR_TERMS.search(text))


def _classify_stratum_b(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if _is_keyword_stuffer(record):
        tags.append("keyword_stuffer")
    if _is_tier5_plain_language(record):
        tags.append("tier5_plain_language")
    if _is_behavioral_rescue(record):
        tags.append("behavioral_rescue")
    if _is_consulting_only(record):
        tags.append("consulting_only")
    if _is_cv_speech_robotics(record):
        tags.append("cv_speech_robotics")
    return tags


def load_candidates_dict(candidates_path: Path) -> dict[str, dict[str, Any]]:
    from tracks.instructor.io import iter_candidates_from_path

    records: dict[str, dict[str, Any]] = {}
    for record in iter_candidates_from_path(candidates_path):
        cid = record.get("candidate_id")
        if not cid:
            continue
        records[str(cid)] = record
    return records


def build_stratified_sample(
    records: dict[str, dict[str, Any]],
    *,
    per_stratum: int,
    random_seed: int,
) -> dict[str, list[SampleEntry]]:
    rng = random.Random(random_seed)

    stratum_a_pool: list[str] = []
    stratum_b_buckets: dict[str, list[str]] = defaultdict(list)

    for cid, record in records.items():
        if _has_ai_signal(record):
            stratum_a_pool.append(cid)
        for tag in _classify_stratum_b(record):
            stratum_b_buckets[tag].append(cid)

    selected_ids: set[str] = set()
    result: dict[str, list[SampleEntry]] = {"A": [], "B": [], "C": []}

    # Stratum A
    rng.shuffle(stratum_a_pool)
    for cid in stratum_a_pool:
        if len(result["A"]) >= per_stratum:
            break
        if cid in selected_ids:
            continue
        selected_ids.add(cid)
        result["A"].append(SampleEntry(cid, ["ai_adjacent"], "A"))

    # Stratum B — up to ceil(per_stratum/5) per bucket
    per_bucket = max(1, math.ceil(per_stratum / len(STRATUM_B_BUCKETS)))
    b_count = 0
    for bucket in STRATUM_B_BUCKETS:
        pool = stratum_b_buckets.get(bucket, [])
        rng.shuffle(pool)
        taken = 0
        for cid in pool:
            if b_count >= per_stratum:
                break
            if taken >= per_bucket:
                break
            if cid in selected_ids:
                continue
            selected_ids.add(cid)
            result["B"].append(SampleEntry(cid, [bucket], "B"))
            taken += 1
            b_count += 1
        if b_count >= per_stratum:
            break

    # Stratum C — random control excluding A∪B picks
    all_ids = list(records.keys())
    rng.shuffle(all_ids)
    for cid in all_ids:
        if len(result["C"]) >= per_stratum:
            break
        if cid in selected_ids:
            continue
        selected_ids.add(cid)
        result["C"].append(SampleEntry(cid, ["random_control"], "C"))

    return result


def entries_to_manifest_dict(
    strata: dict[str, list[SampleEntry]],
    *,
    candidates_path: Path,
    per_stratum: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidates_path": str(candidates_path.resolve()),
        "random_seed": random_seed,
        "per_stratum": per_stratum,
        "strata": {
            key: [
                {"candidate_id": e.candidate_id, "tags": e.tags, "stratum": e.stratum}
                for e in entries
            ]
            for key, entries in strata.items()
        },
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def manifest_entries(manifest: dict[str, Any]) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    strata = manifest.get("strata") or {}
    for stratum_key, items in strata.items():
        for item in items:
            entries.append(
                SampleEntry(
                    candidate_id=str(item["candidate_id"]),
                    tags=list(item.get("tags") or []),
                    stratum=str(item.get("stratum") or stratum_key),
                )
            )
    return entries


def build_and_write_manifest(
    candidates_path: Path,
    output_dir: Path,
    *,
    per_stratum: int,
    random_seed: int,
) -> tuple[dict[str, Any], Path]:
    records = load_candidates_dict(candidates_path)
    strata = build_stratified_sample(
        records, per_stratum=per_stratum, random_seed=random_seed
    )
    manifest = entries_to_manifest_dict(
        strata,
        candidates_path=candidates_path,
        per_stratum=per_stratum,
        random_seed=random_seed,
    )
    path = write_manifest(manifest, output_dir)
    return manifest, path


def build_manifest_from_filtered_ids(
    filtered_ids_path: Path,
    candidates_path: Path,
) -> dict[str, Any]:
    with open(filtered_ids_path, encoding="utf-8") as f:
        candidate_ids = json.load(f)
    if not isinstance(candidate_ids, list):
        raise ValueError(f"Expected JSON array in {filtered_ids_path}")

    entries = [
        SampleEntry(str(cid), ["kmeans_filtered"], "filtered") for cid in candidate_ids
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidates_path": str(candidates_path.resolve()),
        "filtered_ids_path": str(filtered_ids_path.resolve()),
        "sample_mode": "filtered_ids",
        "per_stratum": None,
        "random_seed": None,
        "strata": {
            "filtered": [
                {
                    "candidate_id": e.candidate_id,
                    "tags": e.tags,
                    "stratum": e.stratum,
                }
                for e in entries
            ]
        },
    }


def build_and_write_manifest_from_filtered_ids(
    filtered_ids_path: Path,
    candidates_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = build_manifest_from_filtered_ids(filtered_ids_path, candidates_path)
    path = write_manifest(manifest, output_dir)
    return manifest, path
