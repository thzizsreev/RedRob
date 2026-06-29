"""Compose submission reasoning strings from score decomposition."""

from __future__ import annotations

_REASONING_MAX_LEN = 500
_ABSOLUTE_FALLBACK = (
    "Candidate ranked on composite of retrieval, career history, and availability signals."
)


def _clause1(row: dict) -> str:
    borda = row.get("borda_primary")
    if borda is not None and float(borda) >= 0.75:
        return (
            "Strong technical fit across retrieval, career depth, and JD alignment signals."
        )

    if row.get("in_sweet_spot") is True and float(row.get("tier2_scaled") or 0) > 0:
        years = row.get("total_years_exp")
        if years is not None:
            return (
                f"Within the 6–8 year target range with product-company background. "
                f"{int(float(years))}y total experience."
            )
        return "Within the 6–8 year target range with product-company background."

    ce = row.get("cross_encoder_score")
    if ce is not None and float(ce) >= 0.80:
        return "Closely matches role requirements on full-profile semantic evaluation."

    return "Moderate technical signal across retrieval and career history dimensions."


def _clause2(row: dict) -> str:
    summary = str(row.get("technical_summary_sentence") or "").strip()
    if summary:
        return summary

    if row.get("in_sweet_spot") is True:
        return "Experience within the 6–8 year target range."

    years = row.get("total_years_exp")
    if years is not None:
        return f"{int(float(years))}y total experience."

    return "Profile details not available."


def _clause3(row: dict) -> str | None:
    avail_tier = str(row.get("avail_tier") or "B")
    days = row.get("days_since_active")
    interview = row.get("interview_completion_rate")
    notice = row.get("notice_period_days")
    location = str(row.get("location_tier") or "")
    career = str(row.get("career_type") or "")
    chase_pen = float(row.get("title_chasing_penalty") or row.get("chase_pen") or 0)
    closed_pen = float(row.get("closed_source_penalty") or row.get("closed_pen") or 0)
    short_hops = row.get("short_hop_count")

    if avail_tier == "C" and days is not None and int(days) > 180:
        return f"Note: inactive on platform for {int(days)} days — availability uncertain."

    if (
        avail_tier == "C"
        and interview is not None
        and float(interview) < 0.30
    ):
        pct = int(float(interview) * 100)
        return f"Note: low interview completion rate ({pct}%) — engagement concern."

    if notice is not None and int(notice) > 90:
        return f"Notice period {int(notice)} days raises the hiring bar per JD."

    if location == "outside_india":
        return "Located outside India — no visa sponsorship per JD."

    if chase_pen > 0.06 and short_hops is not None:
        return f"Title-chasing pattern noted: {int(short_hops)} short tenures in history."

    if closed_pen > 0:
        return "Closed-source background with limited external validation signals."

    if career == "mixed":
        return "Mixed product and consulting background — product-company fit is partial."

    return None


def compose_reasoning(row: dict) -> tuple[str, str, str, str | None]:
    c1 = _clause1(row)
    c2 = _clause2(row)
    c3 = _clause3(row)
    parts = [c1, c2]
    if c3:
        parts.append(c3)
    text = " ".join(parts).strip()
    if not text:
        text = _ABSOLUTE_FALLBACK
    return c1, c2, c3, text[:_REASONING_MAX_LEN]
