"""
TerraVision AI — Dashboard section.

Interactive Plotly charts (hover, zoom, pan, click-legend-to-isolate) over the
alerts from the latest scan, plus live filters that drive every chart at once.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from locator import render_location_explorer
from alert_tools import (
    CATEGORY_STYLE,
    LIFECYCLES,
    SEVERITY_COLOR,
    SEVERITY_ORDER,
    SHAPE_FEATURES,
    VERDICT_COLOR,
    VERDICT_ORDER,
    category_color,
    category_label,
    to_dataframe,
)


def _show(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure full-width across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:  # older Streamlit without width="stretch"
        st.plotly_chart(fig, use_container_width=True, key=key)


def _style(fig: go.Figure, is_dark: bool, height: int = 380) -> go.Figure:
    """Apply the TerraVision look: transparent background, theme-matched text and grid."""
    import theme
    fig.update_layout(**theme.plotly_layout(is_dark), margin=dict(l=10, r=10, t=50, b=10), height=height)
    return fig


def _kpi_cards(df: pd.DataFrame, n_resolved: int) -> None:
    n = len(df)
    n_enc = int((df["category"] == "illegal_encroachment").sum())
    n_strong = int((df["verdict"] == "Strong evidence").sum())
    n_new = int((df["lifecycle"] == "new").sum())
    avg_conf = int(round(df["confidence"].mean())) if n else 0
    n_open = int((~df["status"].isin(["verified", "false positive", "closed"])).sum())

    cards = [
        (n, "Alerts in view", "#3b82f6"),
        (n_enc, "Illegal encroachments", "#ff4d6d"),
        (n_strong, "Strong evidence", "#34d399"),
        (n_new, "New this scan", "#fbbf24"),
        (n_resolved, "Resolved since last scan", "#8aa0c4"),
        (n_open, "Awaiting review", "#a78bfa"),
        (f"{avg_conf}%", "Avg confidence", "#22d3ee"),
    ]
    import theme
    st.markdown(theme.kpi_html(cards, min_width=150), unsafe_allow_html=True)


def _filters(df: pd.DataFrame, ns: str, context_ok: List[str]) -> pd.DataFrame:
    """Dashboard filters — return the filtered frame. `ns` resets widget state when the data changes."""
    with st.expander("🎛️  Filter the dashboard", expanded=True):
        c1, c2, c3 = st.columns([2, 2, 2])
        cats = [c for c in CATEGORY_STYLE if c in set(df["category"])]
        cats += [c for c in sorted(set(df["category"])) if c not in cats]
        sel_cats = c1.multiselect(
            "Alert type", cats, default=cats, format_func=category_label, key=f"dash_{ns}_cats"
        )
        sevs = [s for s in SEVERITY_ORDER if s in set(df["severity"])]
        sel_sevs = c2.multiselect("Severity", sevs, default=sevs, key=f"dash_{ns}_sevs")
        regions = sorted(df["region"].unique())
        sel_regions = c3.multiselect("Region", regions, default=regions, key=f"dash_{ns}_regions")

        c4, c5, c6 = st.columns(3)
        verdicts = [v for v in VERDICT_ORDER if v in set(df["verdict"])]
        sel_verdicts = c4.multiselect("Evidence verdict", verdicts, default=verdicts,
                                      key=f"dash_{ns}_verdicts") if verdicts else []
        lifes = [l for l in LIFECYCLES if l in set(df["lifecycle"])]
        sel_life = c5.multiselect("Lifecycle", lifes, default=lifes, key=f"dash_{ns}_life") if lifes else []
        min_conf = c6.slider("Minimum confidence (%)", 0, 100, 0, key=f"dash_{ns}_conf")

        c7, c8, c9 = st.columns(3)
        hide_urban = c7.checkbox("Hide urban lights", value=True, key=f"dash_{ns}_urban",
                                 disabled="land_cover" not in context_ok,
                                 help="Needs the land-cover cross-check to have run for this scan.")
        pa_only = c8.checkbox("Forest & protected areas only", value=True, key=f"dash_{ns}_pa",
                              disabled="protected_area" not in context_ok,
                              help="Needs the protected-area cross-check to have run for this scan.")
        min_cluster = c9.slider("Minimum cluster size (px)", 1, max(int(df["cluster_size"].max()), 2), 1,
                                key=f"dash_{ns}_cluster")

    mask = (
        df["category"].isin(sel_cats) & df["severity"].isin(sel_sevs) & df["region"].isin(sel_regions)
        & (df["confidence"] >= min_conf) & (df["cluster_size"] >= min_cluster)
    )
    if verdicts:
        mask &= df["verdict"].isin(sel_verdicts)
    if lifes:
        mask &= df["lifecycle"].isin(sel_life)
    if hide_urban and "land_cover" in context_ok:
        mask &= ~df["urban_light"]
    if pa_only and "protected_area" in context_ok:
        mask &= df["zone"].ne("outside")
    return df[mask]


# ── individual charts ────────────────────────────────────────────────────────

def _chart_category_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["category"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=[category_label(c) for c in counts.index],
            values=counts.values,
            hole=0.58,
            marker=dict(colors=[category_color(c) for c in counts.index]),
            textinfo="percent",
            hovertemplate="%{label}<br>%{value} alerts (%{percent})<extra></extra>",
            sort=False,
        )
    )
    fig.update_layout(title="Alerts by type")
    return fig


def _chart_severity_bar(df: pd.DataFrame) -> go.Figure:
    ct = df.groupby(["severity", "category"]).size().reset_index(name="count")
    ct["type"] = ct["category"].map(category_label)
    fig = px.bar(
        ct, x="severity", y="count", color="type",
        category_orders={"severity": SEVERITY_ORDER},
        color_discrete_map={category_label(k): v["color"] for k, v in CATEGORY_STYLE.items()},
        labels={"severity": "Severity", "count": "Alerts", "type": "Type"},
        title="Severity breakdown (stacked by type)",
    )
    fig.update_layout(barmode="stack", legend_title_text="")
    return fig


def _chart_confidence_hist(df: pd.DataFrame) -> go.Figure:
    d = df.assign(type=df["category"].map(category_label))
    fig = px.histogram(
        d, x="confidence", color="type", nbins=20,
        color_discrete_map={category_label(k): v["color"] for k, v in CATEGORY_STYLE.items()},
        labels={"confidence": "Confidence (%)", "type": "Type"},
        title="Model confidence distribution",
    )
    fig.update_layout(barmode="stack", yaxis_title="Alerts", legend_title_text="")
    return fig


def _chart_size_vs_radiance(df: pd.DataFrame) -> go.Figure:
    d = df.assign(type=df["category"].map(category_label))
    fig = px.scatter(
        d, x="cluster_size", y="delta_mean", color="type", size="delta_max",
        size_max=22, opacity=0.75, log_x=True,
        color_discrete_map={category_label(k): v["color"] for k, v in CATEGORY_STYLE.items()},
        hover_data={"alert_id": True, "severity": True, "confidence": True,
                    "delta_max": ":.1f", "delta_mean": ":.2f", "type": False},
        labels={"cluster_size": "Cluster size (pixels, log)", "delta_mean": "Mean Δ radiance",
                "type": "Type"},
        title="Cluster size vs. radiance change (bubble = peak Δ)",
    )
    fig.update_layout(legend_title_text="")
    return fig


def _chart_top_alerts(df: pd.DataFrame, n: int = 12) -> go.Figure:
    top = df.nlargest(n, "delta_max").iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["delta_max"], y=top["alert_id"], orientation="h",
            marker=dict(color=[SEVERITY_COLOR.get(s, "#8aa0c4") for s in top["severity"]]),
            customdata=top[["category_label", "severity", "confidence", "cluster_size"]],
            hovertemplate="<b>%{y}</b><br>%{customdata[0]} · %{customdata[1]}"
                          "<br>Peak Δ %{x:.1f} · %{customdata[2]}% conf · %{customdata[3]} px<extra></extra>",
        )
    )
    fig.update_layout(title=f"Top {len(top)} alerts by peak radiance (colour = severity)",
                      xaxis_title="Peak Δ radiance (nW/cm²/sr)")
    return fig


def _chart_radiance_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        fig.add_trace(go.Box(
            y=sub["delta_mean"], name=category_label(cat), boxpoints="outliers",
            marker_color=category_color(cat), line=dict(color=category_color(cat)),
        ))
    fig.update_layout(title="Mean Δ radiance by type", yaxis_title="Mean Δ radiance", showlegend=False)
    return fig


def _chart_shape_radar(df: pd.DataFrame) -> go.Figure:
    cols = [f"shape_{k}" for k in SHAPE_FEATURES if f"shape_{k}" in df.columns]
    fig = go.Figure()
    labels = [c.replace("shape_", "").replace("_", " ") for c in cols]
    for cat in df["category"].unique():
        vals = df.loc[df["category"] == cat, cols].mean().fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], name=category_label(cat),
            line=dict(color=category_color(cat)), fill="toself", opacity=0.35,
        ))
    fig.update_layout(
        title="Average cluster shape fingerprint by type",
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], showticklabels=False)),
    )
    return fig


def _chart_score_heatmap(df: pd.DataFrame) -> go.Figure:
    score_cols = [c for c in df.columns if c.startswith("score_")]
    mat = df.groupby("category")[score_cols].mean()
    fig = px.imshow(
        mat.values,
        x=[c.replace("score_", "").replace("_", " ") for c in score_cols],
        y=[category_label(c) for c in mat.index],
        text_auto=".2f", aspect="auto", color_continuous_scale="Tealgrn",
        labels=dict(x="Class score", y="Predicted type", color="Mean score"),
        title="Classifier scores: predicted type vs. class probability",
    )
    return fig


def _chart_verdict(df: pd.DataFrame) -> go.Figure:
    counts = df["verdict"].replace("", "No cross-check").value_counts()
    colors = [VERDICT_COLOR.get(v, "#546e7a") for v in counts.index]
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=colors,
                           hovertemplate="%{x}: %{y} alerts<extra></extra>"))
    fig.update_layout(title="Evidence verdict (how well other satellites back each alert)",
                      yaxis_title="Alerts")
    return fig


def _chart_lifecycle(df: pd.DataFrame) -> go.Figure:
    d = df.assign(lc=df["lifecycle"].replace("", "unknown"), type=df["category"].map(category_label))
    ct = d.groupby(["lc", "type"]).size().reset_index(name="count")
    fig = px.bar(ct, x="lc", y="count", color="type", labels={"lc": "Lifecycle", "count": "Alerts", "type": "Type"},
                 color_discrete_map={category_label(k): v["color"] for k, v in CATEGORY_STYLE.items()},
                 title="New vs. recurring alerts")
    fig.update_layout(legend_title_text="")
    return fig


def _chart_trend(history: pd.DataFrame, regions: List[str]) -> go.Figure | None:
    """Alerts found by each scan over time (needs at least two scans of a region)."""
    if history is None or history.empty:
        return None
    h = history[history["region"].isin(regions)].copy()
    per_scan = h.groupby(["scan_id", "region", "scanned_at"], as_index=False)["n"].sum()
    if per_scan.groupby("region")["scan_id"].nunique().max() < 2:
        return None
    per_scan["scanned_at"] = pd.to_datetime(per_scan["scanned_at"], errors="coerce")
    fig = px.line(per_scan.sort_values("scanned_at"), x="scanned_at", y="n", color="region", markers=True,
                  labels={"scanned_at": "Scan date", "n": "Alerts", "region": "Region"},
                  title="Alerts found per scan")
    fig.update_layout(legend_title_text="")
    return fig


def _chart_signals(df: pd.DataFrame) -> go.Figure | None:
    """How many alerts each independent signal confirms or contradicts."""
    rows = []
    if df["zone"].notna().any():
        rows.append(("Inside boundary / protected area", int(df["zone"].isin(["official", "protected"]).sum()), "#34d399"))
        rows.append(("Forest outside designated areas", int((df["zone"] == "forest").sum()), "#66bb6a"))
        rows.append(("Outside the forest boundary", int((df["zone"] == "outside").sum()), "#8aa0c4"))
    if df["builtup_fraction"].notna().any():
        rows.append(("Urban light (≥30 % built-up)", int(df["urban_light"].sum()), "#ef5350"))
    if df["ndvi_change"].notna().any():
        rows.append(("Vegetation dropped (NDVI ≤ −0.08)", int((df["ndvi_change"] <= -0.08).sum()), "#66bb6a"))
    if df["tree_loss_fraction"].notna().any():
        rows.append(("Forest loss ≥ 2 % of pixel", int((df["tree_loss_fraction"] >= 0.02).sum()), "#8d6e63"))
    if df["fire_detected"].notna().any():
        rows.append(("Active fire nearby", int((df["fire_detected"] == True).sum()), "#ff8a3d"))        # noqa: E712
    if (df["times_seen"] >= 2).any():
        rows.append(("Seen in 2+ satellite periods", int((df["times_seen"] >= 2).sum()), "#a78bfa"))
    if not rows:
        return None
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
                           marker_color=[r[2] for r in rows],
                           hovertemplate="%{y}: %{x} alerts<extra></extra>"))
    fig.update_layout(title="Independent signals behind the alerts", xaxis_title="Alerts",
                      yaxis=dict(autorange="reversed"))
    return fig


def _chart_confidence_shift(df: pd.DataFrame) -> go.Figure | None:
    """Model confidence vs. evidence-adjusted confidence — points off the diagonal were changed by evidence."""
    d = df[df["verdict"] != ""]
    if d.empty:
        return None
    d = d.assign(type=d["category"].map(category_label))
    fig = px.scatter(
        d, x="model_confidence", y="confidence", color="type", opacity=0.7,
        color_discrete_map={category_label(k): v["color"] for k, v in CATEGORY_STYLE.items()},
        hover_data={"alert_id": True, "verdict": True, "type": False},
        labels={"model_confidence": "Model confidence (%)", "confidence": "Evidence-adjusted (%)", "type": "Type"},
        title="How cross-checks moved each alert's confidence",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dot", color="#8aa0c4"))
    fig.update_layout(legend_title_text="")
    return fig


# ── page ─────────────────────────────────────────────────────────────────────

def render_dashboard(src: Dict, is_dark: bool, history: pd.DataFrame | None = None) -> None:
    """`src` = {label, alerts, resolved, notes, context_ok} as built by app._alert_source()."""
    alerts: List[Dict] = src.get("alerts", [])
    st.markdown('<div class="section-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.caption(f"Data source: {src.get('label', '')}")

    if not alerts:
        st.info("No scan results yet. Open **Monitor** and run a satellite scan — "
                "the dashboard fills in from real scan results.")
        return

    for note in dict.fromkeys(src.get("notes", [])):
        st.info(note)

    context_ok = src.get("context_ok", [])
    full = to_dataframe(alerts)
    ns = str(abs(hash((src.get("label"), len(alerts)))) % 10**8)
    df = _filters(full, ns, context_ok)
    shown_ids = set(df["alert_id"])
    resolved = src.get("resolved", [])
    _kpi_cards(df, len(resolved))

    hidden = len(full) - len(df)
    if hidden:
        st.caption(f"{hidden} of {len(full)} alerts hidden by the filters above.")
    if df.empty:
        st.warning("No alerts match the current filters — widen them above.")
        return

    tab_over, tab_ev, tab_dist, tab_geo, tab_model = st.tabs(
        ["Overview", "Evidence & tracking", "Distributions", "Map", "Classifier insight"]
    )

    with tab_over:
        c1, c2 = st.columns(2)
        with c1:
            _show(_style(_chart_category_donut(df), is_dark), "ch_donut")
        with c2:
            _show(_style(_chart_severity_bar(df), is_dark), "ch_sev")
        _show(_style(_chart_top_alerts(df), is_dark, 420), "ch_top")

    with tab_ev:
        c1, c2 = st.columns(2)
        with c1:
            _show(_style(_chart_verdict(df), is_dark), "ch_verdict")
        with c2:
            _show(_style(_chart_lifecycle(df), is_dark), "ch_life")
        signals = _chart_signals(df)
        if signals is not None:
            _show(_style(signals, is_dark, 340), "ch_signals")
        else:
            st.info("No cross-check evidence for this selection. Re-run the scan with "
                    "“Cross-check with other satellites” enabled.")
        shift = _chart_confidence_shift(df)
        if shift is not None:
            _show(_style(shift, is_dark, 420), "ch_shift")
        trend = _chart_trend(history, sorted(df["region"].unique()))
        if trend is not None:
            _show(_style(trend, is_dark, 340), "ch_trend")
        else:
            st.caption("The trend chart appears once a region has been scanned at least twice.")
        if resolved:
            with st.expander(f"✔️  {len(resolved)} alert(s) no longer glowing since the previous scan"):
                st.dataframe(pd.DataFrame(resolved), hide_index=True)

    with tab_dist:
        c1, c2 = st.columns(2)
        with c1:
            _show(_style(_chart_confidence_hist(df), is_dark), "ch_conf")
        with c2:
            _show(_style(_chart_radiance_box(df), is_dark), "ch_box")
        _show(_style(_chart_size_vs_radiance(df), is_dark, 460), "ch_scatter")

    with tab_geo:
        st.caption("Real satellite imagery. Every pin sits at the alert's exact detected coordinates — "
                   "click a pin for details, or pick an alert to zoom in on its 500 m pixel.")
        render_location_explorer(df.reset_index(drop=True), key="dash_loc")

    with tab_model:
        c1, c2 = st.columns(2)
        with c1:
            _show(_style(_chart_shape_radar(df), is_dark, 420), "ch_radar")
        with c2:
            if any(c.startswith("score_") for c in df.columns):
                _show(_style(_chart_score_heatmap(df), is_dark, 420), "ch_heat")
        st.caption("The shape fingerprint shows how each predicted type differs in cluster geometry; "
                   "the heatmap shows how strongly the classifier separated the four classes.")

    st.markdown(
        "<div style='font-size:0.78rem;color:var(--text-muted);margin-top:8px;'>"
        "Need these alerts in a file, in someone's inbox or on Telegram? Open the <b>Alerts</b> section.</div>",
        unsafe_allow_html=True,
    )
