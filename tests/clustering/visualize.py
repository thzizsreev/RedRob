"""Stage 5 — Plotly scatter plots for cluster and landmark views."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PLOT_DIV_ID = "cluster-scatter-plot"


def _top_skills(record: dict, limit: int = 5) -> str:
    skills = record.get("skills", {})
    names: list[str] = []
    if isinstance(skills, list):
        for item in skills[:limit]:
            if isinstance(item, dict):
                names.append(str(item.get("name", item)))
            else:
                names.append(str(item))
    elif isinstance(skills, dict):
        for key in ("technical", "tools", "soft"):
            for item in skills.get(key, [])[:limit]:
                if isinstance(item, dict):
                    names.append(str(item.get("name", item)))
                else:
                    names.append(str(item))
    return ", ".join(names[:limit]) if names else "(none)"


def _cluster_colors(labels: np.ndarray) -> tuple[list[str], dict[str, str]]:
    labels_str = [str(int(label)) for label in labels]
    unique = sorted(set(labels_str), key=lambda value: (value == "-1", value))
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    color_map = {
        label: ("#9ca3af" if label == "-1" else palette[i % len(palette)])
        for i, label in enumerate(unique)
    }
    return [color_map[label] for label in labels_str], color_map


def _cluster_label_text(cluster: str) -> str:
    return "Noise (unclustered)" if cluster == "-1" else f"Cluster {cluster}"


def _hover_template() -> str:
    return (
        "<b>%{text}</b><br>"
        "Cluster: %{customdata[0]}<br>"
        "Title: %{customdata[1]}<br>"
        "Years: %{customdata[2]}<br>"
        "Skills: %{customdata[3]}<extra></extra>"
    )


def _build_candidate_lookup(
    coords_2d: np.ndarray,
    candidate_ids: list[str],
    records: list[dict],
    labels: np.ndarray,
    color_map: dict[str, str],
) -> dict[str, dict]:
    x_span = float(np.ptp(coords_2d[:, 0])) or 2.0
    y_span = float(np.ptp(coords_2d[:, 1])) or 2.0
    zoom_pad_x = max(x_span * 0.15, 0.8)
    zoom_pad_y = max(y_span * 0.15, 0.8)
    ring_x = max(x_span * 0.012, 0.15)
    ring_y = max(y_span * 0.012, 0.15)

    lookup: dict[str, dict] = {}
    for idx, candidate_id in enumerate(candidate_ids):
        profile = records[idx].get("profile", {})
        cluster = str(int(labels[idx]))
        lookup[candidate_id.upper()] = {
            "id": candidate_id,
            "x": float(coords_2d[idx, 0]),
            "y": float(coords_2d[idx, 1]),
            "cluster": cluster,
            "cluster_label": _cluster_label_text(cluster),
            "cluster_color": color_map.get(cluster, "#dc2626"),
            "title": profile.get("current_title", "(unknown)"),
            "years": profile.get("years_of_experience", "?"),
            "skills": _top_skills(records[idx]),
            "zoom_pad_x": zoom_pad_x,
            "zoom_pad_y": zoom_pad_y,
            "ring_x": ring_x,
            "ring_y": ring_y,
        }
    return lookup


def _build_base_figure(
    coords_2d: np.ndarray,
    candidate_ids: list[str],
    records: list[dict],
    labels: np.ndarray,
    *,
    title: str,
    marker_size: int = 7,
    marker_symbols: list[str] | None = None,
) -> go.Figure:
    profile = [record.get("profile", {}) for record in records]
    cluster_labels = [_cluster_label_text(str(int(label))) for label in labels]
    colors, _color_map = _cluster_colors(labels)
    customdata = np.stack(
        [
            cluster_labels,
            [p.get("current_title", "(unknown)") for p in profile],
            [p.get("years_of_experience", "?") for p in profile],
            [_top_skills(record) for record in records],
        ],
        axis=1,
    )

    marker: dict = {
        "size": marker_size,
        "color": colors,
        "opacity": 0.85,
        "line": {"width": 0.5, "color": "white"},
    }
    if marker_symbols is not None:
        marker["symbol"] = marker_symbols

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode="markers",
            name="candidates",
            text=candidate_ids,
            customdata=customdata,
            hovertemplate=_hover_template(),
            marker=marker,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Selected",
            marker={
                "size": 22,
                "color": "#dc2626",
                "symbol": "circle-open",
                "line": {"width": 4, "color": "#dc2626"},
            },
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title_text="",
        hovermode="closest",
        dragmode="zoom",
    )
    return fig


def _search_post_script(lookup_b64: str, plot_div_id: str) -> str:
    return f"""
(function () {{
  var LOOKUP = JSON.parse(atob("{lookup_b64}"));
  var plotId = "{plot_div_id}";
  var currentHit = null;
  var currentZoom = 1.0;

  function getPlot() {{
    return document.getElementById(plotId);
  }}

  function waitForPlot(callback) {{
    var plot = getPlot();
    if (plot && plot.data && plot.data.length >= 2) {{
      callback(plot);
      return;
    }}
    setTimeout(function () {{ waitForPlot(callback); }}, 50);
  }}

  waitForPlot(function (plot) {{
    var wrap = document.createElement("div");
    wrap.style.cssText = "padding:12px 16px;font-family:sans-serif;background:#f8fafc;border-bottom:1px solid #e5e7eb;";
    wrap.innerHTML =
      '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px;">' +
        '<label for="cid-search"><b>Find candidate:</b></label>' +
        '<input id="cid-search" type="text" placeholder="CAND_0038208" style="padding:6px 10px;min-width:220px;border:1px solid #cbd5e1;border-radius:4px;">' +
        '<button id="cid-search-btn" type="button" style="padding:6px 12px;cursor:pointer;border:1px solid #cbd5e1;border-radius:4px;background:white;">Search</button>' +
        '<button id="cid-zoom-btn" type="button" style="padding:6px 12px;cursor:pointer;border:1px solid #cbd5e1;border-radius:4px;background:white;">Zoom to candidate</button>' +
        '<button id="cid-zoom-in-btn" type="button" style="padding:6px 12px;cursor:pointer;border:1px solid #cbd5e1;border-radius:4px;background:white;">Zoom in</button>' +
        '<button id="cid-zoom-out-btn" type="button" style="padding:6px 12px;cursor:pointer;border:1px solid #cbd5e1;border-radius:4px;background:white;">Zoom out</button>' +
        '<button id="cid-clear-btn" type="button" style="padding:6px 12px;cursor:pointer;border:1px solid #cbd5e1;border-radius:4px;background:white;">Reset view</button>' +
      '</div>' +
      '<div id="cid-info" style="padding:10px 12px;border:1px solid #dbeafe;border-radius:6px;background:#eff6ff;color:#1e3a8a;min-height:24px;">' +
        'Search for a candidate ID to highlight them and show cluster membership.' +
      '</div>' +
      '<div id="cid-search-msg" style="margin-top:8px;color:#475569;"></div>';
    plot.parentNode.insertBefore(wrap, plot);

    function selectionShape(hit) {{
      return [{{
        type: "circle",
        xref: "x",
        yref: "y",
        x0: hit.x - hit.ring_x,
        y0: hit.y - hit.ring_y,
        x1: hit.x + hit.ring_x,
        y1: hit.y + hit.ring_y,
        line: {{ color: "#dc2626", width: 4 }},
        fillcolor: "rgba(220,38,38,0.12)",
        layer: "above",
      }}];
    }}

    function axisRange(hit, scale) {{
      scale = scale || 1.0;
      return {{
        xaxis: {{ range: [hit.x - hit.zoom_pad_x / scale, hit.x + hit.zoom_pad_x / scale] }},
        yaxis: {{ range: [hit.y - hit.zoom_pad_y / scale, hit.y + hit.zoom_pad_y / scale] }},
      }};
    }}

    function renderSelection(hit, scale) {{
      currentHit = hit;
      currentZoom = scale || 1.0;
      var plot = getPlot();
      var ranges = axisRange(hit, currentZoom);
      Plotly.restyle(plot, {{
        x: [[hit.x]],
        y: [[hit.y]],
        "marker.size": [[24]],
        "marker.color": [[hit.cluster_color]],
        "marker.symbol": [["circle-open"]],
        "marker.line.width": [[4]],
        "marker.line.color": [["#dc2626"]],
      }}, [1]);
      Plotly.relayout(plot, {{
        shapes: selectionShape(hit),
        "xaxis.range": ranges.xaxis.range,
        "yaxis.range": ranges.yaxis.range,
      }});
      document.getElementById("cid-info").innerHTML =
        '<b>' + hit.id + '</b><br>' +
        '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:' + hit.cluster_color + ';margin-right:6px;vertical-align:middle;"></span>' +
        '<b>' + hit.cluster_label + '</b><br>' +
        'Title: ' + hit.title + '<br>' +
        'Years: ' + hit.years + '<br>' +
        'Skills: ' + hit.skills;
      document.getElementById("cid-search-msg").textContent =
        "Highlighted at UMAP (" + hit.x.toFixed(3) + ", " + hit.y.toFixed(3) + ")";
      if (typeof Plotly.Fx !== "undefined" && Plotly.Fx.hover) {{
        Plotly.Fx.hover(plot, [{{ curveNumber: 0, pointNumber: hit.index }}]);
      }}
    }}

    function findCandidate(raw) {{
      var q = (raw || "").trim().toUpperCase();
      var msg = document.getElementById("cid-search-msg");
      if (!q) {{
        msg.textContent = "Enter a candidate ID.";
        return;
      }}
      var hit = LOOKUP[q];
      if (!hit) {{
        msg.textContent = "Not found in this sample: " + q;
        document.getElementById("cid-info").innerHTML =
          "No match for <b>" + q + "</b> in this clustering run.";
        Plotly.restyle(getPlot(), {{ x: [[null]], y: [[null]] }}, [1]);
        Plotly.relayout(getPlot(), {{ shapes: [] }});
        currentHit = null;
        return;
      }}
      renderSelection(hit, 1.0);
    }}

    function clearSearch() {{
      currentHit = null;
      currentZoom = 1.0;
      document.getElementById("cid-search").value = "";
      document.getElementById("cid-search-msg").textContent = "";
      document.getElementById("cid-info").innerHTML =
        "Search for a candidate ID to highlight them and show cluster membership.";
      Plotly.restyle(getPlot(), {{ x: [[null]], y: [[null]] }}, [1]);
      Plotly.relayout(getPlot(), {{
        shapes: [],
        "xaxis.autorange": true,
        "yaxis.autorange": true,
      }});
    }}

    document.getElementById("cid-search-btn").onclick = function () {{
      findCandidate(document.getElementById("cid-search").value);
    }};
    document.getElementById("cid-zoom-btn").onclick = function () {{
      if (!currentHit) {{
        findCandidate(document.getElementById("cid-search").value);
        return;
      }}
      renderSelection(currentHit, 1.0);
    }};
    document.getElementById("cid-zoom-in-btn").onclick = function () {{
      if (!currentHit) return;
      currentZoom = Math.min(currentZoom * 1.5, 8.0);
      renderSelection(currentHit, currentZoom);
    }};
    document.getElementById("cid-zoom-out-btn").onclick = function () {{
      if (!currentHit) return;
      currentZoom = Math.max(currentZoom / 1.5, 0.4);
      renderSelection(currentHit, currentZoom);
    }};
    document.getElementById("cid-clear-btn").onclick = clearSearch;
    document.getElementById("cid-search").addEventListener("keydown", function (event) {{
      if (event.key === "Enter") findCandidate(event.target.value);
      if (event.key === "Escape") clearSearch();
    }});
  }});
}})();
"""


def _write_plot_html(
    fig: go.Figure,
    output_path: Path,
    lookup: dict[str, dict],
    *,
    enable_id_search: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    post_script = None
    if enable_id_search:
        lookup_b64 = base64.b64encode(
            json.dumps(lookup, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        post_script = _search_post_script(lookup_b64, PLOT_DIV_ID)

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        div_id=PLOT_DIV_ID,
        post_script=post_script,
    )


def write_cluster_plot(
    coords_2d: np.ndarray,
    candidate_ids: list[str],
    records: list[dict],
    labels: np.ndarray,
    output_path: Path,
    *,
    enable_id_search: bool = True,
) -> None:
    colors, color_map = _cluster_colors(labels)
    del colors
    lookup = _build_candidate_lookup(
        coords_2d, candidate_ids, records, labels, color_map
    )
    for index, candidate_id in enumerate(candidate_ids):
        lookup[candidate_id.upper()]["index"] = index

    fig = _build_base_figure(
        coords_2d,
        candidate_ids,
        records,
        labels,
        title="UMAP 2D — HDBSCAN clusters",
    )
    _write_plot_html(fig, output_path, lookup, enable_id_search=enable_id_search)


def write_landmark_plot(
    coords_2d: np.ndarray,
    candidate_ids: list[str],
    records: list[dict],
    labels: np.ndarray,
    landmark_ids: list[str],
    output_path: Path,
    *,
    enable_id_search: bool = True,
) -> None:
    _, color_map = _cluster_colors(labels)
    lookup = _build_candidate_lookup(
        coords_2d, candidate_ids, records, labels, color_map
    )
    for index, candidate_id in enumerate(candidate_ids):
        lookup[candidate_id.upper()]["index"] = index

    landmark_set = set(landmark_ids)
    marker_symbols = [
        "star" if candidate_id in landmark_set else "circle"
        for candidate_id in candidate_ids
    ]
    fig = _build_base_figure(
        coords_2d,
        candidate_ids,
        records,
        labels,
        title="UMAP 2D — landmarks highlighted",
        marker_size=8,
        marker_symbols=marker_symbols,
    )
    _write_plot_html(fig, output_path, lookup, enable_id_search=enable_id_search)
