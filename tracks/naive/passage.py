"""Build a single passage string from summary + career descriptions."""


def build_passage(record: dict) -> str:
    parts: list[str] = []

    summary = record.get("profile", {}).get("summary", "").strip()
    if summary:
        parts.append(summary)

    for role in record.get("career_history", []):
        title = role.get("title", "").strip()
        company = role.get("company", "").strip()
        desc = role.get("description", "").strip()
        if not desc:
            continue
        prefix = f"{title} at {company}" if company else title
        parts.append(f"{prefix}: {desc}")

    return "\n\n".join(parts)
