"""Random cluster members for manual theme inspection."""

from __future__ import annotations

import random

import numpy as np


def _profile_snippet(record: dict) -> dict:
    profile = record.get("profile", {})
    skills = record.get("skills", {})
    skill_names: list[str] = []
    if isinstance(skills, list):
        for item in skills[:5]:
            if isinstance(item, dict):
                skill_names.append(str(item.get("name", item)))
            else:
                skill_names.append(str(item))
    elif isinstance(skills, dict):
        for key in ("technical", "tools"):
            for item in skills.get(key, [])[:5]:
                if isinstance(item, dict):
                    skill_names.append(str(item.get("name", item)))
                else:
                    skill_names.append(str(item))

    history = record.get("career_history", [])
    latest_desc = ""
    if history:
        latest_desc = str(history[0].get("description", ""))[:240]

    return {
        "candidate_id": record["candidate_id"],
        "current_title": profile.get("current_title", ""),
        "years_of_experience": profile.get("years_of_experience"),
        "top_skills": skill_names[:5],
        "summary_excerpt": str(profile.get("summary", ""))[:240],
        "latest_role_excerpt": latest_desc,
    }


def build_cluster_inspection(
    candidate_ids: list[str],
    records: list[dict],
    labels: np.ndarray,
    random_seed: int,
    *,
    min_members: int = 5,
    max_members: int = 8,
) -> dict:
    del candidate_ids
    rng = random.Random(random_seed)
    clusters: dict[str, dict] = {}

    unique_labels = sorted({int(label) for label in labels})
    for cluster_id in unique_labels:
        member_indices = [i for i, label in enumerate(labels) if int(label) == cluster_id]
        sample_size = min(max_members, max(min_members, len(member_indices)))
        sample_size = min(sample_size, len(member_indices))
        chosen = rng.sample(member_indices, k=sample_size)
        members = [_profile_snippet(records[i]) for i in chosen]
        clusters[str(cluster_id)] = {
            "cluster_id": cluster_id,
            "size": len(member_indices),
            "theme": "",
            "sample_members": members,
        }

    return {
        "instructions": (
            "Fill in the one-line theme per cluster after reading sample_members."
        ),
        "clusters": clusters,
    }
