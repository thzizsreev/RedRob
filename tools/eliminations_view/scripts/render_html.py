"""Self-contained HTML viewer for elimination records."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_html(dataset: dict[str, Any], *, title: str = "Pipeline Eliminations") -> str:
    meta = dataset["meta"]
    eliminations = dataset["eliminations"]
    built = html.escape(str(meta.get("built_at_utc", "")))
    count = meta.get("elimination_count", len(eliminations))
    counts_by_stage = meta.get("counts_by_stage", {})
    stage_summary = ", ".join(f"S{k}: {v}" for k, v in sorted(counts_by_stage.items()))

    payload = json.dumps(dataset, ensure_ascii=False)
    payload_safe = payload.replace("</", "<\\/")

    reasons = sorted({e["elimination"]["reason_code"] for e in eliminations})
    reason_labels = {
        e["elimination"]["reason_code"]: e["elimination"].get("reason_label", e["elimination"]["reason_code"])
        for e in eliminations
    }
    reason_options = "".join(
        f'<option value="{html.escape(r)}">{html.escape(reason_labels.get(r, r))}</option>'
        for r in reasons
    )
    categories = sorted({e["elimination"]["category"] for e in eliminations})
    category_labels = {
        "honeypot": "Honeypot",
        "experience": "Experience band",
        "title": "Title gate",
        "consulting": "Consulting only",
        "research": "Research only",
        "shallow_ai": "Shallow AI",
        "retrieval": "Retrieval cut",
        "rerank": "Rerank cut",
        "final_score": "Final score cut",
        "gate": "Other gate",
    }
    category_options = "".join(
        f'<option value="{html.escape(c)}">{html.escape(category_labels.get(c, c))}</option>'
        for c in categories
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
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
      align-items: center;
    }}
    #search, select {{
      padding: .6rem .9rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: .95rem;
    }}
    #search {{ flex: 1; min-width: 220px; }}
    select {{ min-width: 140px; }}
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
      grid-template-columns: 72px 1fr 24px;
      gap: 1rem;
      align-items: center;
      padding: 1rem 1.1rem;
      cursor: pointer;
      user-select: none;
    }}
    .card-summary::-webkit-details-marker {{ display: none; }}
    .badge-col {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: .25rem;
    }}
    .stage-badge {{
      font-size: .72rem;
      font-weight: 700;
      letter-spacing: .04em;
      padding: .2rem .45rem;
      border-radius: 6px;
      background: var(--surface2);
      border: 1px solid var(--border);
    }}
    .cat-badge {{
      font-size: .62rem;
      text-transform: uppercase;
      letter-spacing: .04em;
      padding: .2rem .4rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      text-align: center;
      max-width: 72px;
      line-height: 1.25;
    }}
    .cat-hint {{
      font-size: .58rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-top: .15rem;
    }}
    .name {{ font-size: 1.05rem; font-weight: 600; }}
    .headline {{ color: var(--accent); font-size: .92rem; margin-top: .15rem; }}
    .meta-line {{ color: var(--muted); font-size: .82rem; margin-top: .2rem; }}
    .meta-line code {{ font-size: .78rem; color: #a8b8cc; }}
    .summary-line {{
      margin-top: .45rem;
      font-size: .86rem;
      color: #b8c5d6;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .card[open] .summary-line {{ -webkit-line-clamp: unset; }}
    .chevron {{ color: var(--muted); font-size: .7rem; transition: transform .2s; text-align: center; }}
    .card[open] .chevron {{ transform: rotate(180deg); }}
    .card-body {{
      padding: 0 1.1rem 1.25rem;
      border-top: 1px solid var(--border);
    }}
    section {{ margin-top: 1.1rem; }}
    section h3 {{
      margin: 0 0 .5rem;
      font-size: .78rem;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
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
    #stats {{ font-size: .85rem; color: var(--muted); }}
    #empty {{ text-align: center; color: var(--muted); padding: 3rem 1rem; display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Built {built} · {count} eliminated · {html.escape(stage_summary)}</p>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search by CAND id, name, headline…" autocomplete="off">
      <select id="stage-filter" aria-label="Filter by stage">
        <option value="">All stages</option>
        <option value="2">Stage 2</option>
        <option value="3">Stage 3</option>
        <option value="4">Stage 4</option>
        <option value="5">Stage 5</option>
      </select>
      <select id="category-filter" aria-label="Filter by category">
        <option value="">All categories</option>
        {category_options}
      </select>
      <select id="reason-filter" aria-label="Filter by reason">
        <option value="">All reasons</option>
        {reason_options}
      </select>
      <button class="btn" id="expand-all" type="button">Expand visible</button>
      <button class="btn" id="collapse-all" type="button">Collapse all</button>
      <span id="stats"></span>
    </div>
  </header>
  <main>
    <div id="empty">No records match your filters.</div>
    <div id="list"></div>
  </main>
  <script type="application/json" id="dataset">{payload_safe}</script>
  <script>
    const CATEGORY_COLORS = {{
      honeypot: '#e07a5f',
      experience: '#f2cc8f',
      title: '#d4a574',
      consulting: '#c9a86c',
      research: '#a8b8cc',
      shallow_ai: '#9b8ec4',
      retrieval: '#81b29a',
      rerank: '#5b9fd4',
      final_score: '#b8a9c9',
      gate: '#8b9cb3',
    }};

    const dataset = JSON.parse(document.getElementById('dataset').textContent);
    const entries = dataset.eliminations || [];
    const listEl = document.getElementById('list');
    const searchEl = document.getElementById('search');
    const stageEl = document.getElementById('stage-filter');
    const categoryEl = document.getElementById('category-filter');
    const reasonEl = document.getElementById('reason-filter');
    const statsEl = document.getElementById('stats');
    const emptyEl = document.getElementById('empty');
    const cardEls = [];
    const bodyCache = new Map();

    function esc(s) {{
      if (s === null || s === undefined) return '';
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }}

    function fmtNum(v) {{
      if (v === null || v === undefined) return '—';
      if (typeof v === 'boolean') return v ? 'yes' : 'no';
      if (typeof v === 'number') {{
        if (Math.abs(v) >= 100 || (v > 0 && v < 0.001)) return v.toPrecision(4);
        return String(Math.round(v * 1000) / 1000);
      }}
      return esc(v);
    }}

    function kvTable(data) {{
      if (!data || typeof data !== 'object') return '<p class="muted">No data</p>';
      const rows = [];
      for (const [key, value] of Object.entries(data)) {{
        if (value === null || value === '') continue;
        const label = esc(key.replace(/_/g, ' '));
        let cell;
        if (typeof value === 'object') {{
          cell = '<pre>' + esc(JSON.stringify(value, null, 2)) + '</pre>';
        }} else {{
          cell = fmtNum(value);
        }}
        rows.push('<tr><th>' + label + '</th><td>' + cell + '</td></tr>');
      }}
      return rows.length ? '<table class="kv">' + rows.join('') + '</table>' : '<p class="muted">No data</p>';
    }}

    function renderSkills(skills) {{
      if (!skills || !skills.length) return '<p class="muted">No skills listed</p>';
      const tags = skills.map(skill => {{
        if (typeof skill !== 'object') return '<span class="tag">' + esc(skill) + '</span>';
        const name = skill.name || skill.skill || '';
        const parts = [];
        if (skill.proficiency) parts.push(skill.proficiency);
        if (skill.duration_months != null) parts.push(skill.duration_months + 'mo');
        const label = parts.length ? name + ' (' + parts.join(', ') + ')' : name;
        return '<span class="tag">' + esc(label) + '</span>';
      }});
      return '<div class="tags">' + tags.join('') + '</div>';
    }}

    function renderTimeline(items, titleKey) {{
      titleKey = titleKey || 'title';
      if (!items || !items.length) return '<p class="muted">None</p>';
      const blocks = items.map(item => {{
        if (!item || typeof item !== 'object') return '';
        const heading = esc(item[titleKey] || item.degree || 'Entry');
        const org = esc(item.company || item.institution || '');
        let dates = '';
        const start = item.start_date || '';
        const end = item.end_date || (item.is_current ? 'Present' : '');
        if (start || end) dates = '<span class="dates">' + esc(start) + ' – ' + esc(end) + '</span>';
        const desc = item.description || item.field_of_study || '';
        const descHtml = desc ? '<p>' + esc(desc) + '</p>' : '';
        return '<article class="timeline-item"><h4>' + heading + (org ? ' @ ' + org : '') + '</h4>' + dates + descHtml + '</article>';
      }});
      return '<div class="timeline">' + blocks.join('') + '</div>';
    }}

    function renderBody(entry) {{
      const profile = entry.profile || {{}};
      const elimination = entry.elimination || {{}};
      const pipeline = entry.pipeline || {{}};
      const details = elimination.details || {{}};
      const summary = profile.summary || '';
      return `
        <section><h3>Elimination</h3>${{kvTable({{
          reason_code: elimination.reason_code,
          reason_label: elimination.reason_label,
          badge_label: elimination.badge_label,
          category: elimination.category,
          summary: elimination.summary,
          rules: elimination.rules,
        }})}}</section>
        <section><h3>Rule details</h3>${{Object.keys(details).length ? kvTable(details) : '<p class="muted">None</p>'}}</section>
        <section><h3>Pipeline scores</h3>${{Object.keys(pipeline).length ? kvTable(pipeline) : '<p class="muted">None</p>'}}</section>
        <section><h3>Professional summary</h3><p>${{summary ? esc(summary) : '<span class="muted">No summary</span>'}}</p></section>
        <section><h3>Career history</h3>${{renderTimeline(entry.career_history)}}</section>
        <section><h3>Education</h3>${{renderTimeline(entry.education, 'degree')}}</section>
        <section><h3>Skills</h3>${{renderSkills(entry.skills)}}</section>
      `;
    }}

    function profileName(entry) {{
      return (entry.profile && entry.profile.anonymized_name) || entry.candidate_id;
    }}

    function buildCard(entry, index) {{
      const el = entry.elimination || {{}};
      const profile = entry.profile || {{}};
      const cid = entry.candidate_id;
      const name = profileName(entry);
      const headline = profile.headline || '';
      const stage = entry.stage;
      const category = el.category || '';
      const badgeLabel = el.badge_label || el.reason_label || category || 'Unknown';
      const catColor = CATEGORY_COLORS[category] || '#8b9cb3';
      const details = document.createElement('details');
      details.className = 'card';
      details.dataset.id = cid.toLowerCase();
      details.dataset.name = name.toLowerCase();
      details.dataset.headline = headline.toLowerCase();
      details.dataset.stage = String(stage);
      details.dataset.category = category;
      details.dataset.reason = el.reason_code || '';
      details.dataset.index = String(index);

      details.innerHTML = `
        <summary class="card-summary">
          <div class="badge-col">
            <span class="stage-badge">S${{stage}}</span>
            <span class="cat-badge" style="color:${{catColor}};border-color:${{catColor}}" title="${{esc(el.reason_label || '')}}">${{esc(badgeLabel)}}</span>
            <span class="cat-hint">${{esc(category)}}</span>
          </div>
          <div class="main-col">
            <div class="name">${{esc(name)}}</div>
            <div class="headline">${{esc(headline)}}</div>
            <div class="meta-line"><code>${{esc(cid)}}</code> · ${{esc(el.reason_label || el.reason_code)}}</div>
            <div class="summary-line">${{esc(el.summary)}}</div>
          </div>
          <div class="chevron" aria-hidden="true">▼</div>
        </summary>
        <div class="card-body" data-loaded="false"></div>
      `;

      details.addEventListener('toggle', () => {{
        if (!details.open) return;
        const body = details.querySelector('.card-body');
        if (body.dataset.loaded === 'true') return;
        const idx = Number(details.dataset.index);
        if (!bodyCache.has(idx)) bodyCache.set(idx, renderBody(entries[idx]));
        body.innerHTML = bodyCache.get(idx);
        body.dataset.loaded = 'true';
      }});

      return details;
    }}

    for (let i = 0; i < entries.length; i++) {{
      const card = buildCard(entries[i], i);
      listEl.appendChild(card);
      cardEls.push(card);
    }}

    function applyFilters() {{
      const q = searchEl.value.trim().toLowerCase();
      const stage = stageEl.value;
      const category = categoryEl.value;
      const reason = reasonEl.value;
      let visible = 0;
      for (const card of cardEls) {{
        const hay = (card.dataset.id + ' ' + card.dataset.name + ' ' + card.dataset.headline).toLowerCase();
        const show = (!q || hay.includes(q))
          && (!stage || card.dataset.stage === stage)
          && (!category || card.dataset.category === category)
          && (!reason || card.dataset.reason === reason);
        card.classList.toggle('hidden', !show);
        if (show) visible++;
      }}
      statsEl.textContent = visible === cardEls.length
        ? `${{cardEls.length}} shown`
        : `${{visible}} / ${{cardEls.length}} shown`;
      emptyEl.style.display = visible === 0 ? 'block' : 'none';
    }}

    searchEl.addEventListener('input', applyFilters);
    stageEl.addEventListener('change', applyFilters);
    categoryEl.addEventListener('change', applyFilters);
    reasonEl.addEventListener('change', applyFilters);

    document.getElementById('expand-all').addEventListener('click', () => {{
      cardEls.forEach(c => {{ if (!c.classList.contains('hidden')) c.open = true; }});
    }});
    document.getElementById('collapse-all').addEventListener('click', () => {{
      cardEls.forEach(c => c.open = false);
    }});

    applyFilters();
  </script>
</body>
</html>
"""


def write_html_files(dataset: dict[str, Any], out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}

    all_html = out_dir / "all_eliminations.html"
    all_html.write_text(
        render_html(dataset, title="Pipeline Eliminations (Stages 2–5)"),
        encoding="utf-8",
    )
    paths["all_html"] = str(all_html.resolve())

    for stage, rows in dataset.get("by_stage", {}).items():
        stage_dir = out_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_dataset = {
            "meta": {**dataset["meta"], "stage": stage, "elimination_count": len(rows)},
            "eliminations": rows,
        }
        stage_html = stage_dir / "eliminations.html"
        stage_html.write_text(
            render_html(stage_dataset, title=f"Stage {stage} Eliminations"),
            encoding="utf-8",
        )
        paths[f"stage{stage}_html"] = str(stage_html.resolve())

    return paths
