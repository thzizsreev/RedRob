"""Compose submission reasoning strings from score decomposition."""

from __future__ import annotations


def _strength_clause(row: dict) -> str:
    ce = row.get("cross_encoder_score")
    q1 = row.get("q1_score")
    parts: list[str] = []
    if ce is not None:
        parts.append(f"cross-encoder {float(ce):.2f}")
    if q1 is not None:
        parts.append(f"Q1 retrieval fit {float(q1):.3f}")
    if row.get("in_sweet_spot"):
        parts.append("experience in JD sweet spot")
    if float(row.get("product_company_fraction") or 0) > 0.6:
        parts.append("product-company career mix")
    if not parts:
        return "Solid composite relevance across retrieval and ranking signals"
    return "Strong fit on " + ", ".join(parts[:3])


def _concern_clause(row: dict) -> str | None:
    concerns: list[tuple[float, str]] = []

    floor = float(row.get("must_have_floor_multiplier") or 1.0)
    if floor < 0.7:
        concerns.append((0.7 - floor, f"thin must-have coverage (floor {floor:.2f})"))

    if row.get("stale_coding"):
        concerns.append((0.15, "no hands-on coding in the last 18 months"))

    closed = float(row.get("closed_source_penalty") or 0.0)
    if closed > 0:
        concerns.append((closed, "limited external validation or GitHub presence"))

    avail = float(row.get("availability_multiplier") or 1.0)
    if avail < 0.9:
        concerns.append((1.0 - avail, f"availability dampener ({avail:.2f})"))

    if row.get("location_tier") == "outside_india":
        concerns.append((0.10, "located outside India (no visa sponsorship)"))

    notice = float(row.get("notice_adj") or 0.0)
    if notice < 0:
        concerns.append((abs(notice), "notice period above 30 days"))

    if row.get("career_type") == "consulting_heavy":
        concerns.append((0.04, "consulting-heavy career mix"))

    q3p = float(row.get("q3_residual_penalty") or 0.0)
    if q3p > 0.05:
        concerns.append((q3p, "anti-pattern similarity in career profile"))

    if not concerns:
        return None
    concerns.sort(key=lambda x: -x[0])
    return f"Concern: {concerns[0][1]}."


def compose_reasoning(row: dict) -> str:
    anchor = str(row.get("technical_summary_sentence") or "").strip()
    strength = _strength_clause(row)
    concern = _concern_clause(row)

    if anchor:
        text = f"{anchor} {strength}."
    else:
        text = f"{strength}."

    if concern:
        text = f"{text.rstrip('.')}. {concern}"
    return text[:500]
