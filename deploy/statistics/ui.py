"""Statistics UI launcher and Streamlit app (Phase 4) — Enhanced Dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from share.config.config_loader import load_config
from share.kernel.statistics.sqlite_store import (
    get_class_totals,
    get_overview,
    get_recent_events,
    init_stats_db,
)
from share.types.errors import ConfigError

# ---------------------------------------------------------------------------
# Custom CSS — Modern Light Theme (Glassmorphism & Animations)
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #f4f6f8; /* Soft light gray background */
    --bg-card: #ffffff;    /* Pure white for cards */
    --border: rgba(0, 0, 0, 0.08);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --text-primary: #0f172a; /* Slate 900 */
    --text-muted: #64748b;   /* Slate 500 */
    --radius: 16px;
}

/* Base App Setup */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}

header[data-testid="stHeader"] { background: transparent !important; }

/* Headings with Gradient */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #0f172a 0%, var(--accent-blue) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.5rem !important;
}

h2, h3, h4, h5, .stSubheader {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

/* Glassmorphism Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--bg-card) 0%, rgba(255,255,255,0.7) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(59, 130, 246, 0.12);
    border-color: rgba(59, 130, 246, 0.3);
}

[data-testid="stMetricLabel"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 8px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-size: 2.2rem !important;
}

/* DataFrames */
[data-testid="stDataFrame"], .stDataFrame {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    overflow: hidden;
    background: var(--bg-card);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(15px) !important;
    border-right: 1px solid var(--border) !important;
}

/* Modern Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    padding: 0px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 12px 20px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* Status Pill with Pulse Animation */
@keyframes pulse-green {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.3); }
    70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.05em;
    backdrop-filter: blur(4px);
}
.status-pill.online {
    background: rgba(16,185,129,0.1);
    color: #059669;
    border: 1px solid rgba(16,185,129,0.3);
    animation: pulse-green 2.5s infinite;
}
.status-pill.empty {
    background: rgba(249,115,22,0.1);
    color: #ea580c;
    border: 1px solid rgba(249,115,22,0.3);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: currentColor;
}

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }
</style>
"""

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4 statistics UI")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Config override, repeatable.",
    )
    parser.add_argument("--db-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--event-limit", type=int, default=200, help=argparse.SUPPRESS)
    return parser


# ---------------------------------------------------------------------------
# Plotly chart builders - Light Theme Visuals
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#475569"),
    hoverlabel=dict(
        bgcolor="rgba(255, 255, 255, 0.95)", 
        font=dict(family="JetBrains Mono", size=12, color="#0f172a"),
        bordercolor="rgba(0,0,0,0.1)"
    ),
)

_AXIS_STYLE = dict(
    showgrid=True, 
    gridcolor="rgba(0,0,0,0.05)",
    zeroline=False,
    tickfont=dict(size=11, color="#64748b", family="JetBrains Mono")
)

def _try_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


def _make_latency_chart(events, go):
    if not events:
        return None
    ts = [e["ts_utc"] for e in reversed(events)]
    lat = [e["latency_ms"] for e in reversed(events)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=lat, mode="lines", fill="tozeroy",
        line=dict(color="#3b82f6", width=3, shape="spline"), # Smooth curves
        fillcolor="rgba(59, 130, 246, 0.15)",
        hovertemplate="<b>%{y:.1f} ms</b><br>%{x}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT_BASE, margin=dict(l=0, r=0, t=20, b=10), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(**_AXIS_STYLE, title=dict(text="Latency (ms)", font=dict(size=11))),
    )
    return fig


def _make_detections_chart(events, go):
    if not events:
        return None
    ts = [e["ts_utc"] for e in reversed(events)]
    det = [e["total_detections"] for e in reversed(events)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ts, y=det,
        marker=dict(
            color=det, 
            colorscale=[[0, "rgba(6,182,212,0.4)"], [1, "rgba(6,182,212,1)"]],
            line=dict(width=0), 
            cornerradius=4
        ),
        hovertemplate="<b>%{y} detections</b><br>%{x}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT_BASE, margin=dict(l=0, r=0, t=20, b=10), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(**_AXIS_STYLE, title=dict(text="Count", font=dict(size=11))),
    )
    return fig


def _make_class_donut(class_totals, go):
    if not class_totals:
        return None
    labels = list(class_totals.keys())
    values = list(class_totals.values())
    palette = ["#3b82f6", "#06b6d4", "#10b981", "#f97316", "#8b5cf6",
               "#ec4899", "#eab308", "#34d399", "#6366f1", "#ef4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.7,
        marker=dict(colors=palette[:len(labels)], line=dict(color="#ffffff", width=3)), # White gaps
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=12, family="Outfit", color="#475569"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )])
    fig.update_layout(
        **_PLOTLY_LAYOUT_BASE, margin=dict(l=20, r=20, t=20, b=20), height=340,
        showlegend=True,
        legend=dict(
            font=dict(family="Outfit", size=12, color="#0f172a"), 
            bgcolor="rgba(0,0,0,0)",
            orientation="v",
            yanchor="middle", y=0.5
        ),
    )
    return fig


def _make_source_bar(events, go):
    if not events:
        return None
    src_cnt: dict[str, int] = {}
    src_det: dict[str, int] = {}
    for e in events:
        sid = e["source_id"]
        src_cnt[sid] = src_cnt.get(sid, 0) + 1
        src_det[sid] = src_det.get(sid, 0) + e["total_detections"]

    sources = sorted(src_cnt, key=lambda s: src_cnt[s], reverse=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sources, x=[src_cnt[s] for s in sources], orientation="h",
        name="Events", marker=dict(color="#3b82f6", cornerradius=6),
        hovertemplate="<b>%{y}</b><br>Events: %{x}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=sources, x=[src_det[s] for s in sources], orientation="h",
        name="Detections", marker=dict(color="rgba(6,182,212,0.6)", cornerradius=6),
        hovertemplate="<b>%{y}</b><br>Detections: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT_BASE, margin=dict(l=0, r=0, t=20, b=10),
        height=max(220, len(sources) * 50 + 60), 
        barmode="group", bargap=0.2, bargroupgap=0.1,
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#0f172a", family="JetBrains Mono")),
        legend=dict(
            font=dict(family="Outfit", size=12),
            bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1
        ),
    )
    return fig


def _aggregate_class_totals(events: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for event in events:
        counts = event.get("counts_by_class", {})
        if not isinstance(counts, dict):
            continue
        for name, count in counts.items():
            try:
                c = int(count)
            except (TypeError, ValueError):
                continue
            totals[str(name)] = totals.get(str(name), 0) + c
    return totals


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * max(0.0, min(100.0, p)) / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def _build_filtered_summary(events: list[dict[str, Any]]) -> dict[str, float]:
    if not events:
        return {
            "events_total": 0.0,
            "detections_total": 0.0,
            "detections_per_event": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    lats = [float(e["latency_ms"]) for e in events]
    detections_total = float(sum(int(e["total_detections"]) for e in events))
    events_total = float(len(events))
    return {
        "events_total": events_total,
        "detections_total": detections_total,
        "detections_per_event": detections_total / max(events_total, 1.0),
        "latency_p50_ms": _percentile(lats, 50),
        "latency_p95_ms": _percentile(lats, 95),
        "latency_max_ms": max(lats),
    }


def _build_source_detail_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        source = str(event["source_id"])
        entry = grouped.setdefault(
            source,
            {
                "source_id": source,
                "events": 0,
                "detections": 0,
                "latencies": [],
            },
        )
        entry["events"] += 1
        entry["detections"] += int(event["total_detections"])
        entry["latencies"].append(float(event["latency_ms"]))

    rows: list[dict[str, Any]] = []
    for source, raw in sorted(grouped.items()):
        lats = raw["latencies"]
        events_total = int(raw["events"])
        detections_total = int(raw["detections"])
        rows.append(
            {
                "Source ID": source,
                "Events": events_total,
                "Detections": detections_total,
                "Detections/Event": round(detections_total / max(events_total, 1), 3),
                "Latency P50 (ms)": round(_percentile(lats, 50), 3),
                "Latency P95 (ms)": round(_percentile(lats, 95), 3),
                "Latency Max (ms)": round(max(lats) if lats else 0.0, 3),
            }
        )
    return rows


def _build_class_detail_rows(class_totals: dict[str, int]) -> list[dict[str, Any]]:
    if not class_totals:
        return []
    total = max(sum(class_totals.values()), 1)
    rows: list[dict[str, Any]] = []
    for class_name, count in sorted(class_totals.items(), key=lambda x: x[1], reverse=True):
        rows.append(
            {
                "Class": class_name,
                "Count": int(count),
                "Share (%)": round(float(count) * 100.0 / total, 2),
            }
        )
    return rows


def _make_latency_histogram(events, go):
    if not events:
        return None
    lats = [float(e["latency_ms"]) for e in events]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=lats,
            nbinsx=min(30, max(8, len(lats) // 2)),
            marker=dict(color="rgba(59,130,246,0.75)"),
            hovertemplate="<b>%{x:.2f} ms</b><br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT_BASE,
        margin=dict(l=0, r=0, t=20, b=10),
        height=260,
        xaxis=dict(**_AXIS_STYLE, title=dict(text="Latency (ms)", font=dict(size=11))),
        yaxis=dict(**_AXIS_STYLE, title=dict(text="Events", font=dict(size=11))),
    )
    return fig


# ---------------------------------------------------------------------------
# Main Streamlit renderer
# ---------------------------------------------------------------------------

def _render_streamlit_ui(db_path: Path, event_limit: int) -> int:
    try:
        import streamlit as st
    except Exception as exc:
        print(f"[RUNTIME ERROR] streamlit is required: {exc}", file=sys.stderr)
        return 3

    go = _try_import_plotly()
    init_stats_db(db_path)
    overview = get_overview(db_path)
    recent = get_recent_events(db_path, limit=event_limit)
    class_totals_all = get_class_totals(db_path, limit=event_limit)
    source_options = sorted({str(e["source_id"]) for e in recent})
    max_detection_seen = max((int(e["total_detections"]) for e in recent), default=0)

    # ── Page config ──
    st.set_page_config(
        page_title="Vision Refactor · Analytics",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        auto_refresh = st.toggle("🔄 Auto Refresh", value=False)
        refresh_sec = st.selectbox("⏱️ Refresh Interval (sec)", [5, 10, 20, 30], index=1)
        selected_sources = st.multiselect(
            "📡 Source Filter",
            options=source_options,
            default=source_options,
        )
        min_detections = st.slider(
            "🎯 Min Detections / Event",
            min_value=0,
            max_value=max(max_detection_seen, 1),
            value=0,
            step=1,
        )
        st.markdown("<hr style='margin: 1rem 0 !important;'>", unsafe_allow_html=True)
        st.markdown("**📁 Database Path**")
        st.code(str(db_path), language=None)
        st.markdown(f"**🎯 Event Limit:** `{event_limit}`")
        if auto_refresh:
            import time as _time

            _time.sleep(int(refresh_sec))
            st.rerun()

    filtered_recent = [
        event
        for event in recent
        if (not selected_sources or str(event["source_id"]) in selected_sources)
        and int(event["total_detections"]) >= min_detections
    ]
    filtered_class_totals = _aggregate_class_totals(filtered_recent)
    filtered_summary = _build_filtered_summary(filtered_recent)

    # ── Header ──
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.title("Vision Refactor Analytics")
        st.markdown("Real-time telemetry and detection monitoring dashboard.")
    with hdr_r:
        has_data = overview["events_total"] > 0
        pill_cls = "online" if has_data else "empty"
        pill_lbl = "SYSTEM ONLINE" if has_data else "WAITING FOR DATA"
        st.markdown(
            f'<div style="height: 100%; display: flex; justify-content: flex-end; align-items: center;">'
            f'<span class="status-pill {pill_cls}"><div class="status-dot"></div>{pill_lbl}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── KPI row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📦 Total Events", f"{overview['events_total']:,}")
    m2.metric("🎯 Total Detections", f"{overview['detections_total']:,}")
    m3.metric("⚡ Avg Latency", f"{overview['avg_latency_ms']:.1f} ms")
    m4.metric("📡 Active Sources", overview["source_count"])
    last_ts = overview["last_event_ts_utc"]
    m5.metric("🕐 Last Update (UTC)", last_ts[11:19] if last_ts else "—")

    st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("🔍 Filtered Events", f"{int(filtered_summary['events_total']):,}")
    f2.metric("📈 Filtered Detections", f"{int(filtered_summary['detections_total']):,}")
    f3.metric("🎯 Detections/Event", f"{filtered_summary['detections_per_event']:.2f}")
    f4.metric("⚡ P95 Latency", f"{filtered_summary['latency_p95_ms']:.1f} ms")

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    if recent and not filtered_recent:
        st.warning("当前过滤条件下没有事件，已展示总体指标。可调整 Source / Min Detections 过滤。")

    # ── Main content ──
    if filtered_recent and go is not None:
        tab_ov, tab_src, tab_log = st.tabs(["📊 Dashboard Overview", "📡 Source Analysis", "📋 Raw Event Log"])

        with tab_ov:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### ⏱️ Latency Trend")
                fig = _make_latency_chart(filtered_recent, go)
                if fig:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="ov_latency",
                    )
            with c2:
                st.markdown("##### 🔍 Detections Trend")
                fig = _make_detections_chart(filtered_recent, go)
                if fig:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="ov_detections",
                    )

            st.markdown("<hr style='margin: 1rem 0 !important;'>", unsafe_allow_html=True)

            d1, d2 = st.columns([1, 1.2])
            with d1:
                st.markdown("##### 🏷️ Object Class Distribution")
                if filtered_class_totals:
                    fig = _make_class_donut(filtered_class_totals, go)
                    if fig:
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="ov_donut",
                        )
                else:
                    st.info("No class data available yet.")
            with d2:
                st.markdown("##### 📈 Top Contributing Sources")
                fig = _make_source_bar(filtered_recent, go)
                if fig:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="ov_source",
                    )

            st.markdown("<hr style='margin: 1rem 0 !important;'>", unsafe_allow_html=True)
            l1, l2 = st.columns([1.4, 1.0])
            with l1:
                st.markdown("##### 📉 Latency Distribution")
                fig = _make_latency_histogram(filtered_recent, go)
                if fig:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="ov_latency_hist",
                    )
            with l2:
                st.markdown("##### 🧮 Latency Statistics")
                stat_rows = [
                    {"Metric": "P50", "Latency (ms)": round(filtered_summary["latency_p50_ms"], 3)},
                    {"Metric": "P95", "Latency (ms)": round(filtered_summary["latency_p95_ms"], 3)},
                    {"Metric": "Max", "Latency (ms)": round(filtered_summary["latency_max_ms"], 3)},
                ]
                st.dataframe(stat_rows, use_container_width=True, hide_index=True)
                st.markdown("##### 🧾 Top Classes")
                st.dataframe(
                    _build_class_detail_rows(filtered_class_totals)[:10],
                    use_container_width=True,
                    hide_index=True,
                    height=240,
                )

        with tab_src:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📊 Events & Detections by Source")
            fig = _make_source_bar(filtered_recent, go)
            if fig:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="src_source",
                )

            source_rows = _build_source_detail_rows(filtered_recent)
            if source_rows:
                st.markdown("##### ⚡ Per-Source Latency Statistics")
                st.dataframe(source_rows, use_container_width=True, hide_index=True)

        with tab_log:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📋 Recent Edge Events")
            rows = []
            for e in filtered_recent:
                cls_str = ", ".join(f"{k}: {v}" for k, v in e["counts_by_class"].items()) or "—"
                rows.append(
                    {
                        "Event ID": e["id"],
                        "Timestamp (UTC)": e["ts_utc"][:19].replace("T", " "),
                        "Source": e["source_id"],
                        "Detections": e["total_detections"],
                        "Latency (ms)": round(e["latency_ms"], 2),
                        "Class Breakdown": cls_str,
                    }
                )
            st.dataframe(rows, use_container_width=True, height=600)

            event_by_id = {int(e["id"]): e for e in filtered_recent}
            selected_event_id = st.selectbox(
                "🔎 Inspect Event (raw payload)",
                options=sorted(event_by_id.keys(), reverse=True),
            )
            st.json(event_by_id[int(selected_event_id)])

    elif filtered_recent:
        # Fallback without plotly
        st.subheader("Class Totals")
        if filtered_class_totals:
            st.bar_chart(filtered_class_totals)
        st.subheader("Recent Events")
        rows = []
        for e in filtered_recent:
            cls_str = ", ".join(f"{k}:{v}" for k, v in e["counts_by_class"].items()) or "—"
            rows.append(
                {
                    "ID": e["id"],
                    "Time": e["ts_utc"][:19].replace("T", " "),
                    "Source": e["source_id"],
                    "Detections": e["total_detections"],
                    "Latency (ms)": round(e["latency_ms"], 2),
                    "Classes": cls_str,
                }
            )
        st.dataframe(rows, use_container_width=True)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("🔌 No events match current filters. Relax filters or push more events.")
        if class_totals_all:
            st.subheader("All Classes (unfiltered window)")
            st.dataframe(
                _build_class_detail_rows(class_totals_all),
                use_container_width=True,
                hide_index=True,
                height=260,
            )

    # ── Footer ──
    st.markdown("---")
    st.caption(
        "💾 Storage: "
        f"`{db_path}` &nbsp; | &nbsp; "
        f"🎯 Limit: `{event_limit}` events &nbsp; | &nbsp; "
        f"🔍 Filtered: `{len(filtered_recent)}` events &nbsp; | &nbsp; "
        "Vision Refactor Statistics v2.1"
    )
    return 0


def _launch_streamlit(cfg: dict[str, Any]) -> int:
    stats_cfg = cfg["deploy"]["statistics"]
    db_path = Path(stats_cfg["db_path"])
    init_stats_db(db_path)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).resolve()),
        "--server.address", str(stats_cfg["public_host"]),
        "--server.port", str(int(stats_cfg["ui_port"])),
        "--", "--db-path", str(db_path), "--event-limit", "200",
    ]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, _unknown = parser.parse_known_args(argv)

    if args.db_path:
        return _render_streamlit_ui(db_path=Path(args.db_path), event_limit=int(args.event_limit))

    if not args.config:
        print("[USAGE ERROR] --config is required when launching UI.", file=sys.stderr)
        return 2

    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None

    try:
        cfg = load_config(
            config_path=config_path, overrides=args.set, workdir_override=workdir_override,
        )
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    return _launch_streamlit(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
