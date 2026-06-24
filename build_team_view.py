#!/usr/bin/env python3
"""
Build pretty JSON + interactive HTML from Stage 5 team submission CSV.

    python build_team_view.py
    python build_team_view.py --out team_results_view

Joins team_xxx.csv with full candidates.jsonl profiles and optional Stage 5
scored parquet for pipeline metadata.

Default output: team_results_view/ in the redrob repo root.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracks.shared.paths import (
    CANDIDATES_JSONL_PATH,
    ROOT_DIR,
    RUNTIME_STAGE5_DIR,
)

DEFAULT_CSV = RUNTIME_STAGE5_DIR / "team_xxx.csv"
DEFAULT_STAGE5_PARQUET = RUNTIME_STAGE5_DIR / "stage5_scored_top100.parquet"
DEFAULT_OUT_DIR = ROOT_DIR / "team_results_view"

PIPELINE_SCORE_KEYS = (
    "stage3_rank",
    "stage4_rank",
    "cross_encoder_score",
    "fused_score",
    "q1_score",
    "q1_rank",
    "q2_score",
    "q2_rank",
    "bm25_score",
    "bm25_rank",
    "rrf_score",
    "q3_neg_sim",
)

STAGE5_SCORE_KEYS = (
    "final_score",
    "ce_norm",
    "fused_norm",
    "q1_norm",
    "q2_norm",
    "q3_norm",
    "core",
    "keyword_ratio",
    "assessment_cov",
    "combined_coverage",
    "must_have_floor_multiplier",
    "core_floored",
    "shape_mult",
    "shaped",
    "title_chasing_penalty",
    "q3_residual_penalty",
    "closed_source_penalty",
    "ambiguity_penalty",
    "consulting_resid_penalty",
    "total_penalty",
    "penalized",
    "optional_bonus",
    "bonused",
    "availability_multiplier",
    "availability_adj",
    "location_adj",
    "workmode_adj",
    "notice_adj",
    "logistics_adjustment",
)

GATE_KEYS = (
    "total_years_exp",
    "exp_band",
    "in_sweet_spot",
    "title_family",
    "skill_kw_density",
    "title_ambiguous",
    "stale_profile",
    "low_responder",
    "not_open",
    "honeypot_anomaly_score",
    "product_company_count",
    "consulting_company_count",
    "product_company_fraction",
    "career_type",
    "research_fraction",
    "research_heavy",
    "has_any_production_role",
    "stale_coding",
    "currently_between_roles",
    "months_since_last_ic_role",
    "pre_llm_production_ml",
    "recent_ai_only",
    "llm_framework_only",
    "ml_experience_start_year",
    "avg_tenure_per_employer",
    "short_hop_count",
    "title_progression_jumps",
    "location_tier",
    "external_validation_score",
    "has_github",
    "notice_period_days",
    "cluster_id",
    "cluster_rank",
    "dist_to_centroid",
)

SIGNAL_KEYS = (
    "open_to_work_flag",
    "last_active_date",
    "applications_submitted_30d",
    "recruiter_response_rate",
    "avg_response_time_hours",
    "interview_completion_rate",
    "offer_acceptance_rate",
    "preferred_work_mode",
    "profile_completeness_score",
    "profile_views_received_30d",
    "saved_by_recruiters_30d",
    "github_activity_score",
    "notice_period_days",
    "expected_salary_range_inr_lpa",
    "willing_to_relocate",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build team results JSON + HTML viewer.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Stage 5 submission CSV")
    parser.add_argument(
        "--stage5-parquet",
        type=Path,
        default=DEFAULT_STAGE5_PARQUET,
        help="Stage 5 scored parquet for pipeline metadata",
    )
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_JSONL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_submission_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission CSV: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "reasoning": row.get("reasoning", "").strip(),
                }
            )
    rows.sort(key=lambda r: r["rank"])
    return rows


def load_pipeline_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    import polars as pl

    df = pl.read_parquet(path)
    meta: dict[str, dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        cid = str(row["candidate_id"])
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if key == "candidate_id":
                continue
            if value is None:
                cleaned[key] = None
            elif hasattr(value, "item"):
                cleaned[key] = value.item()
            else:
                cleaned[key] = value
        meta[cid] = cleaned
    return meta


def stream_candidates(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(wanted)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = str(record.get("candidate_id", ""))
            if cid in remaining:
                found[cid] = record
                remaining.remove(cid)
    return found


def _pick_keys(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: source.get(k) for k in keys if k in source}


def build_record(
    submission: dict[str, Any],
    profile: dict[str, Any],
    pipeline_row: dict[str, Any] | None,
) -> dict[str, Any]:
    pipeline_row = pipeline_row or {}
    signals = profile.get("redrob_signals") or {}
    skill_assessments = pipeline_row.get("skill_assessment_scores") or signals.get(
        "skill_assessment_scores"
    )

    return {
        "candidate_id": submission["candidate_id"],
        "submission": {
            "rank": submission["rank"],
            "score": submission["score"],
            "reasoning": submission["reasoning"],
        },
        "profile": profile.get("profile") or {},
        "career_history": profile.get("career_history") or [],
        "education": profile.get("education") or [],
        "skills": profile.get("skills") or pipeline_row.get("skills") or [],
        "certifications": profile.get("certifications") or [],
        "languages": profile.get("languages") or [],
        "pipeline": {
            "retrieval_scores": _pick_keys(pipeline_row, PIPELINE_SCORE_KEYS),
            "stage5_scoring": _pick_keys(pipeline_row, STAGE5_SCORE_KEYS),
            "gates_and_career": _pick_keys(pipeline_row, GATE_KEYS),
            "behavioral_signals": {
                **_pick_keys(pipeline_row, SIGNAL_KEYS),
                **_pick_keys(signals, SIGNAL_KEYS),
            },
            "skill_assessment_scores": skill_assessments,
            "technical_summary_sentence": pipeline_row.get("technical_summary_sentence"),
            "profile_headline": pipeline_row.get("profile_headline"),
        },
    }


def build_dataset(
    submission_rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    pipeline_meta: dict[str, dict[str, Any]],
    sources: dict[str, str],
) -> dict[str, Any]:
    candidates = []
    for sub in submission_rows:
        cid = sub["candidate_id"]
        candidates.append(
            build_record(sub, profiles[cid], pipeline_meta.get(cid))
        )
    return {
        "meta": {
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "sources": sources,
        },
        "candidates": candidates,
    }


def _fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 100 or (0 < abs(value) < 0.001 and value != 0):
            return f"{value:.4g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return html.escape(str(value))


def _render_kv_table(data: dict[str, Any]) -> str:
    if not data:
        return "<p class='muted'>No data</p>"
    rows = []
    for key, value in data.items():
        if value is None or value == "":
            continue
        label = html.escape(key.replace("_", " "))
        if isinstance(value, (dict, list)):
            cell = f"<pre>{html.escape(json.dumps(value, indent=2, ensure_ascii=False))}</pre>"
        else:
            cell = _fmt_num(value)
        rows.append(f"<tr><th>{label}</th><td>{cell}</td></tr>")
    if not rows:
        return "<p class='muted'>No data</p>"
    return "<table class='kv'>" + "".join(rows) + "</table>"


def _render_skills(skills: list[Any]) -> str:
    if not skills:
        return "<p class='muted'>No skills listed</p>"
    tags = []
    for skill in skills:
        if isinstance(skill, dict):
            name = skill.get("name", skill.get("skill", ""))
            prof = skill.get("proficiency", "")
            label = f"{name} ({prof})" if prof else str(name)
        else:
            label = str(skill)
        tags.append(f"<span class='tag'>{html.escape(label)}</span>")
    return "<div class='tags'>" + "".join(tags) + "</div>"


def _render_timeline(items: list[dict[str, Any]], title_key: str = "title") -> str:
    if not items:
        return "<p class='muted'>None</p>"
    blocks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        heading = html.escape(str(item.get(title_key, item.get("degree", "Entry"))))
        org = html.escape(str(item.get("company", item.get("institution", ""))))
        dates = ""
        start = item.get("start_date", "")
        end = item.get("end_date") or ("Present" if item.get("is_current") else "")
        if start or end:
            dates = f"<span class='dates'>{html.escape(str(start))} – {html.escape(str(end))}</span>"
        desc = item.get("description", item.get("field_of_study", ""))
        desc_html = f"<p>{html.escape(str(desc))}</p>" if desc else ""
        blocks.append(
            f"<article class='timeline-item'><h4>{heading}"
            f"{f' @ {org}' if org else ''}</h4>{dates}{desc_html}</article>"
        )
    return "<div class='timeline'>" + "".join(blocks) + "</div>"


def render_html(dataset: dict[str, Any]) -> str:
    cards = []
    for entry in dataset["candidates"]:
        sub = entry["submission"]
        profile = entry["profile"]
        rank = sub["rank"]
        score = sub["score"]
        cid = entry["candidate_id"]
        name = html.escape(str(profile.get("anonymized_name", cid)))
        headline = html.escape(str(profile.get("headline", "")))
        location = html.escape(
            ", ".join(
                p
                for p in [profile.get("location"), profile.get("country")]
                if p
            )
        )
        reasoning = html.escape(sub.get("reasoning", ""))
        summary = html.escape(str(profile.get("summary", "")))

        pipeline = entry.get("pipeline") or {}
        retrieval = _render_kv_table(pipeline.get("retrieval_scores") or {})
        stage5 = _render_kv_table(pipeline.get("stage5_scoring") or {})
        gates = _render_kv_table(pipeline.get("gates_and_career") or {})
        behavioral = _render_kv_table(pipeline.get("behavioral_signals") or {})
        assessments = pipeline.get("skill_assessment_scores")
        assessments_html = (
            _render_kv_table(assessments)
            if isinstance(assessments, dict)
            else "<p class='muted'>None</p>"
        )

        cards.append(
            f"""
            <details class="card" data-rank="{rank}" data-name="{name.lower()}" data-headline="{headline.lower()}">
              <summary class="card-summary">
                <div class="score-col">
                  <div class="rank">#{rank}</div>
                  <div class="score">{score:.3f}</div>
                </div>
                <div class="main-col">
                  <div class="name">{name}</div>
                  <div class="headline">{headline}</div>
                  <div class="meta-line">{location} · <code>{html.escape(cid)}</code></div>
                  <div class="reasoning">{reasoning}</div>
                </div>
                <div class="chevron" aria-hidden="true">▼</div>
              </summary>
              <div class="card-body">
                <section>
                  <h3>Professional summary</h3>
                  <p>{summary or '<span class="muted">No summary</span>'}</p>
                </section>
                <section>
                  <h3>Career history</h3>
                  {_render_timeline(entry.get("career_history") or [])}
                </section>
                <section>
                  <h3>Education</h3>
                  {_render_timeline(entry.get("education") or [], title_key="degree")}
                </section>
                <section>
                  <h3>Skills</h3>
                  {_render_skills(entry.get("skills") or [])}
                </section>
                <section class="grid-2">
                  <div>
                    <h3>Retrieval &amp; rerank</h3>
                    {retrieval}
                  </div>
                  <div>
                    <h3>Stage 5 scoring breakdown</h3>
                    {stage5}
                  </div>
                </section>
                <section class="grid-2">
                  <div>
                    <h3>Gates &amp; career signals</h3>
                    {gates}
                  </div>
                  <div>
                    <h3>Behavioral signals</h3>
                    {behavioral}
                  </div>
                </section>
                <section>
                  <h3>Skill assessments</h3>
                  {assessments_html}
                </section>
              </div>
            </details>
            """
        )

    count = dataset["meta"]["candidate_count"]
    built = html.escape(dataset["meta"]["built_at_utc"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Team Results — Top {count}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --surface2: #243044;
      --border: #2d3a4f;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #5b9fd4;
      --accent2: #7ccea0;
      --score-bg: linear-gradient(160deg, #1e3a5f 0%, #162447 100%);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      padding: 1.5rem 2rem;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    header h1 {{ margin: 0 0 .25rem; font-size: 1.4rem; font-weight: 600; }}
    header p {{ margin: 0; color: var(--muted); font-size: .9rem; }}
    .toolbar {{
      margin-top: 1rem;
      display: flex;
      gap: .75rem;
      flex-wrap: wrap;
    }}
    #search {{
      flex: 1;
      min-width: 220px;
      padding: .6rem .9rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: .95rem;
    }}
    .btn {{
      padding: .6rem 1rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface2);
      color: var(--text);
      cursor: pointer;
      font-size: .9rem;
    }}
    .btn:hover {{ border-color: var(--accent); }}
    main {{ max-width: 960px; margin: 0 auto; padding: 1rem 1rem 3rem; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: .75rem;
      overflow: hidden;
      transition: border-color .15s;
    }}
    .card[open] {{ border-color: var(--accent); }}
    .card-summary {{
      list-style: none;
      display: grid;
      grid-template-columns: 88px 1fr 24px;
      gap: 1rem;
      align-items: center;
      padding: 1rem 1.1rem;
      cursor: pointer;
      user-select: none;
    }}
    .card-summary::-webkit-details-marker {{ display: none; }}
    .score-col {{
      background: var(--score-bg);
      border-radius: 10px;
      padding: .55rem .4rem;
      text-align: center;
      border: 1px solid #2a4a6b;
    }}
    .rank {{ font-size: .75rem; color: var(--muted); font-weight: 600; letter-spacing: .04em; }}
    .score {{ font-size: 1.35rem; font-weight: 700; color: var(--accent2); line-height: 1.2; }}
    .name {{ font-size: 1.05rem; font-weight: 600; }}
    .headline {{ color: var(--accent); font-size: .92rem; margin-top: .15rem; }}
    .meta-line {{ color: var(--muted); font-size: .82rem; margin-top: .2rem; }}
    .meta-line code {{ font-size: .78rem; color: #a8b8cc; }}
    .reasoning {{
      margin-top: .45rem;
      font-size: .86rem;
      color: #b8c5d6;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .card[open] .reasoning {{ -webkit-line-clamp: unset; }}
    .chevron {{ color: var(--muted); font-size: .7rem; transition: transform .2s; text-align: center; }}
    .card[open] .chevron {{ transform: rotate(180deg); }}
    .card-body {{
      padding: 0 1.1rem 1.25rem;
      border-top: 1px solid var(--border);
      animation: fade .2s ease;
    }}
    @keyframes fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    section {{ margin-top: 1.1rem; }}
    section h3 {{
      margin: 0 0 .5rem;
      font-size: .78rem;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}
    @media (max-width: 720px) {{
      .card-summary {{ grid-template-columns: 72px 1fr 20px; gap: .75rem; }}
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
    .timeline-item {{
      padding: .65rem 0;
      border-bottom: 1px solid var(--border);
    }}
    .timeline-item:last-child {{ border-bottom: none; }}
    .timeline-item h4 {{ margin: 0; font-size: .95rem; }}
    .dates {{ display: block; font-size: .8rem; color: var(--muted); margin: .2rem 0; }}
    .timeline-item p {{ margin: .35rem 0 0; font-size: .88rem; color: #c5d0de; }}
    table.kv {{ width: 100%; border-collapse: collapse; font-size: .84rem; }}
    table.kv th {{
      text-align: left;
      color: var(--muted);
      font-weight: 500;
      padding: .25rem .5rem .25rem 0;
      vertical-align: top;
      width: 42%;
    }}
    table.kv td {{ padding: .25rem 0; word-break: break-word; }}
    table.kv pre {{
      margin: 0;
      font-size: .75rem;
      white-space: pre-wrap;
      color: #c5d0de;
    }}
    .tags {{ display: flex; flex-wrap: wrap; gap: .35rem; }}
    .tag {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: .2rem .55rem;
      font-size: .78rem;
    }}
    .muted {{ color: var(--muted); font-style: italic; }}
    .hidden {{ display: none !important; }}
    #stats {{ font-size: .85rem; color: var(--muted); align-self: center; }}
  </style>
</head>
<body>
  <header>
    <h1>Stage 5 Team Results</h1>
    <p>Built {built} · {count} candidates ranked</p>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search name, headline, ID…" autocomplete="off">
      <button class="btn" id="expand-all" type="button">Expand all</button>
      <button class="btn" id="collapse-all" type="button">Collapse all</button>
      <span id="stats"></span>
    </div>
  </header>
  <main id="list">
    {"".join(cards)}
  </main>
  <script>
    const cards = [...document.querySelectorAll('.card')];
    const search = document.getElementById('search');
    const stats = document.getElementById('stats');
    function updateStats(visible) {{
      stats.textContent = visible === cards.length
        ? `${{cards.length}} shown`
        : `${{visible}} / ${{cards.length}} shown`;
    }}
    function filterCards() {{
      const q = search.value.trim().toLowerCase();
      let visible = 0;
      for (const card of cards) {{
        const hay = (card.dataset.name + ' ' + card.dataset.headline + ' ' + card.dataset.rank).toLowerCase();
        const show = !q || hay.includes(q);
        card.classList.toggle('hidden', !show);
        if (show) visible++;
      }}
      updateStats(visible);
    }}
    search.addEventListener('input', filterCards);
    document.getElementById('expand-all').addEventListener('click', () => {{
      cards.forEach(c => {{ if (!c.classList.contains('hidden')) c.open = true; }});
    }});
    document.getElementById('collapse-all').addEventListener('click', () => {{
      cards.forEach(c => c.open = false);
    }});
    updateStats(cards.length);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_rows = load_submission_csv(args.csv)
    wanted = {r["candidate_id"] for r in submission_rows}
    profiles = stream_candidates(args.candidates, wanted)

    missing = sorted(wanted - set(profiles))
    if missing:
        raise ValueError(
            f"{len(missing)} submission candidate(s) missing from JSONL. "
            f"Examples: {missing[:5]}"
        )

    pipeline_meta = load_pipeline_metadata(args.stage5_parquet)
    sources = {
        "submission_csv": str(args.csv.resolve()),
        "candidates_jsonl": str(args.candidates.resolve()),
        "stage5_parquet": str(args.stage5_parquet.resolve())
        if args.stage5_parquet.exists()
        else None,
    }
    dataset = build_dataset(submission_rows, profiles, pipeline_meta, sources)

    json_path = out_dir / "team_results.json"
    html_path = out_dir / "team_results.html"
    manifest_path = out_dir / "manifest.json"

    json_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_html(dataset), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "built_at_utc": dataset["meta"]["built_at_utc"],
                "candidate_count": dataset["meta"]["candidate_count"],
                "sources": sources,
                "outputs": {
                    "team_results_json": str(json_path),
                    "team_results_html": str(html_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Built {len(submission_rows)} candidate profiles")
    print(f"  {json_path}")
    print(f"  {html_path}")
    print(f"  {manifest_path}")
    print(f"\nOpen in browser: file:///{html_path.as_posix()}")


if __name__ == "__main__":
    main()
