"""
TerraVision AI — exact-location map.

Shows alerts at their precise coordinates on real satellite imagery (Esri World
Imagery) with place labels. A selected alert gets a crosshair marker, the
~500 m VIIRS pixel it was detected in, and copyable coordinates.
"""

from __future__ import annotations

import html

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen, MarkerCluster
from streamlit_folium import st_folium

from alert_tools import (
    PIXEL_SIZE_M,
    category_color,
    category_label,
    format_dms,
    pixel_footprint,
)

IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
STREETS = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ATTR_ESRI = "Imagery © Esri, Maxar, Earthstar Geographics"


def _popup(r: pd.Series) -> str:
    color = category_color(r["category"])
    return (
        "<div style='font-family:sans-serif;font-size:13px;min-width:230px;'>"
        f"<b style='color:{color}'>{category_label(r['category']).upper()}</b> · {html.escape(str(r['alert_id']))}<br>"
        "<hr style='margin:6px 0'>"
        f"📍 <b>{r['lat']:.6f}, {r['lon']:.6f}</b><br>"
        f"<span style='color:#666'>{format_dms(r['lat'], 'N', 'S')} {format_dms(r['lon'], 'E', 'W')}</span><br>"
        f"⚠️ {r['severity'].upper()} · 🎯 {r['confidence']}% confidence<br>"
        f"🔲 {r['cluster_size']} px cluster · ⚡ Δ peak {r['delta_max']}<br>"
        + (f"🧭 <b>{html.escape(str(r.get('verdict')))}</b><br>" if r.get("verdict") else "") +
        f"<a href='{r['maps_url']}' target='_blank'>Open in Google Maps ↗</a>"
        "</div>"
    )


def build_map(df: pd.DataFrame, selected_id: str | None = None) -> folium.Map:
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=6,
                   tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(IMAGERY, attr=ATTR_ESRI, name="Satellite imagery", max_zoom=19).add_to(m)
    folium.TileLayer(STREETS, attr="© OpenStreetMap contributors", name="Street map",
                     show=False, max_zoom=19).add_to(m)
    folium.TileLayer(LABELS, attr=ATTR_ESRI, name="Place labels", overlay=True,
                     max_zoom=19).add_to(m)
    Fullscreen().add_to(m)

    selected = df[df["alert_id"] == selected_id] if selected_id else df.iloc[0:0]
    others = df.drop(selected.index)

    layer = MarkerCluster(name="Alerts", disableClusteringAtZoom=13) if len(others) > 150 \
        else folium.FeatureGroup(name="Alerts")
    for _, r in others.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=6, color="#ffffff", weight=1,
            fill=True, fill_color=category_color(r["category"]), fill_opacity=0.95,
            popup=folium.Popup(_popup(r), max_width=280),
            tooltip=f"{r['alert_id']} · {category_label(r['category'])} · {r['lat']:.5f}, {r['lon']:.5f}",
        ).add_to(layer)
    layer.add_to(m)

    if len(selected):
        r = selected.iloc[0]
        bounds = pixel_footprint(r["lat"], r["lon"])
        color = category_color(r["category"])
        folium.Rectangle(
            bounds, color=color, weight=2, fill=True, fill_opacity=0.18,
            tooltip=f"Detected satellite pixel (~{PIXEL_SIZE_M} m × {PIXEL_SIZE_M} m)",
        ).add_to(m)
        folium.Marker(
            [r["lat"], r["lon"]],
            icon=folium.DivIcon(
                icon_size=(30, 30), icon_anchor=(15, 15),
                html=(f"<div style='width:30px;height:30px;border:3px solid {color};border-radius:50%;"
                      "box-shadow:0 0 0 2px #fff,0 0 12px rgba(0,0,0,.6);position:relative'>"
                      f"<div style='position:absolute;left:12px;top:-8px;width:2px;height:44px;background:{color}'></div>"
                      f"<div style='position:absolute;top:12px;left:-8px;height:2px;width:44px;background:{color}'></div>"
                      "</div>"),
            ),
            popup=folium.Popup(_popup(r), max_width=280, show=True),
            tooltip=f"{r['lat']:.6f}, {r['lon']:.6f}",
        ).add_to(m)
        m.fit_bounds([[bounds[0][0] - 0.004, bounds[0][1] - 0.004],
                      [bounds[1][0] + 0.004, bounds[1][1] + 0.004]])
    else:
        m.fit_bounds([[df["lat"].min(), df["lon"].min()], [df["lat"].max(), df["lon"].max()]])

    folium.LayerControl(collapsed=True).add_to(m)
    return m


def render_location_explorer(df: pd.DataFrame, key: str, height: int = 560) -> None:
    """Alert picker + satellite map + exact coordinates. `df` is the (filtered) alert table."""
    if df.empty:
        st.info("No alerts to place on the map.")
        return

    ids = df["alert_id"].tolist()
    labels = {
        r.alert_id: f"{r.alert_id} · {r.severity.upper()} · {category_label(r.category)} · "
                    f"{r.lat:.5f}, {r.lon:.5f}"
        for r in df.itertuples()
    }
    c_pick, c_all = st.columns([4, 1])
    choice = c_pick.selectbox(
        "Zoom to an alert", ["All alerts"] + ids,
        format_func=lambda i: "Show all alerts" if i == "All alerts" else labels[i],
        key=f"{key}_pick",
    )
    selected_id = None if choice == "All alerts" else choice

    # Large result sets are capped for map speed; the selected alert is always kept.
    shown = df
    if len(df) > 1500:
        shown = df.nlargest(1500, "delta_max")
        if selected_id and selected_id not in set(shown["alert_id"]):
            shown = pd.concat([shown, df[df["alert_id"] == selected_id]])
        c_all.caption(f"Map shows top 1,500 of {len(df)} by radiance")

    left, right = st.columns([3, 1.2]) if selected_id else (st.container(), None)
    with left:
        st_folium(build_map(shown, selected_id), width="100%", height=height,
                  returned_objects=[], key=f"{key}_map_{selected_id or 'all'}")

    if selected_id:
        _imagery_panel(df[df["alert_id"] == selected_id].iloc[0])

    if selected_id and right is not None:
        r = df[df["alert_id"] == selected_id].iloc[0]
        with right:
            st.markdown("**Exact location**")
            st.code(f"{r['lat']:.6f}, {r['lon']:.6f}", language="text")
            st.caption(f"{format_dms(r['lat'], 'N', 'S')}  {format_dms(r['lon'], 'E', 'W')}")
            st.markdown(
                f"**{category_label(r['category'])}** · {r['severity'].upper()}  \n"
                f"{r['cluster_size']} px cluster · {r['confidence']}% confidence  \n"
                f"Region: {r['region']}  \nDetected: {r['detected_at']}"
            )
            if r.get("verdict"):
                st.markdown(f"**Evidence:** {r['verdict']}")
                for reason in [e for e in str(r.get("evidence_text", "")).split("; ") if e]:
                    st.caption(f"• {reason}")
            if r.get("status"):
                st.caption(f"Review status: {r['status']}" + (f" · {r['assignee']}" if r.get("assignee") else ""))
            st.link_button("Open in Google Maps ↗", r["maps_url"])
            st.caption(
                f"The pin is the cluster's centre. VIIRS pixels are ~{PIXEL_SIZE_M} m, so the "
                "outlined square is the area the satellite actually measured — confirm on the ground "
                "within that square."
            )


def _imagery_panel(r: pd.Series) -> None:
    """Optional before/after Sentinel-2 view for the selected alert (fetched on request)."""
    cache = st.session_state.setdefault("imagery_cache", {})
    key = str(r["alert_id"])
    st.markdown("**Look at the ground**")
    if st.button("🛰️  Show before / after satellite imagery", key=f"img_btn_{key}"):
        try:
            with st.spinner("Fetching Sentinel-2 imagery from Earth Engine…"):
                from gee_gateway import connect_gee
                from imagery import before_after_urls
                connect_gee()
                cache[key] = before_after_urls(float(r["lat"]), float(r["lon"]))
        except Exception as e:
            cache[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    result = cache.get(key)
    if not result:
        st.caption("Same place and season, one year apart, with the 500 m satellite pixel outlined in yellow. "
                   "Look for new clearings, tracks, roofs or burn scars inside the square.")
        return
    if "error" in result:
        st.warning(f"Could not load imagery: {result['error']}")
        return
    c1, c2 = st.columns(2)
    for col, k, title in ((c1, "before", "Before"), (c2, "after", "After")):
        with col:
            if result.get(k):
                st.image(result[k], caption=f"{title}: {result['labels'][k]} ({result['scenes'][k]} scenes)")
            else:
                st.info(f"{title}: no cloud-free Sentinel-2 scenes for {result['labels'][k]}.")
