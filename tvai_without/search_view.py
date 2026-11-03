"""
TerraVision AI — Search section.

Type a place name or coordinates: the map jumps there, and the page says whether NASA thermal
satellites see fire nearby, whether night lights look unusual, or that no anomaly was detected.
"""

from __future__ import annotations

import html
from typing import Dict

import folium
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

import geocode
import placecheck
from alert_tools import format_dms

IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
ATTR = "Imagery © Esri, Maxar, Earthstar Geographics"

TONE = {  # status -> (colour, icon)
    placecheck.FIRE: ("#ff4d6d", "🔥"), placecheck.LIGHT: ("#fbbf24", "💡"),
    placecheck.CLEAR: ("#34d399", "✅"), placecheck.UNKNOWN: ("#8aa0c4", "❔"),
}


def _map(place: Dict, result: Dict | None) -> folium.Map:
    lat, lon = place["lat"], place["lon"]
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(IMAGERY, attr=ATTR, name="Satellite imagery", max_zoom=19).add_to(m)
    folium.TileLayer(LABELS, attr=ATTR, name="Place labels", overlay=True, max_zoom=19).add_to(m)
    Fullscreen().add_to(m)
    color = TONE[result["status"]][0] if result else "#7db4ff"
    if result:
        folium.Circle([lat, lon], radius=result["radius_km"] * 1000, color=color, weight=1.5, dash_array="6 6",
                      fill=True, fill_opacity=0.06, tooltip=f"Checked within {result['radius_km']:g} km").add_to(m)
        for h in result["hotspots"]:
            folium.CircleMarker(
                [h["lat"], h["lon"]], radius=7, color="#ffffff", weight=1.5, fill=True, fill_color="#ff4d6d",
                fill_opacity=0.95,
                tooltip=f"🔥 {h['distance_km']:.1f} km away · {h['temp_k']:.0f} K",
            ).add_to(m)
    folium.Marker([lat, lon], tooltip=html.escape(place["name"][:80]),
                  icon=folium.Icon(color="blue", icon="search", prefix="fa")).add_to(m)
    if place.get("bbox") and not result:
        w, s, e, n = place["bbox"]
        m.fit_bounds([[s, w], [n, e]])
    return m


def render_search_page() -> None:
    st.markdown('<div class="section-title">Find a place</div>', unsafe_allow_html=True)
    with st.form("place_form"):
        c1, c2, c3, c4 = st.columns([4, 1.4, 1.4, 1.2])
        query = c1.text_input("Place or coordinates", placeholder="e.g. Sariska Tiger Reserve  ·  27.2046, 77.4977",
                              label_visibility="collapsed")
        radius = c2.selectbox("Radius", [5, 10, 25, 50], index=1, format_func=lambda k: f"{k} km")
        days = c3.selectbox("Fire window", [3, 7, 10, 14], index=2, format_func=lambda k: f"last {k} days")
        go = c4.form_submit_button("🔎  Search")

    if go and query.strip():
        st.session_state.pop("place_result", None)
        try:
            with st.spinner("Finding the place…"):
                found = geocode.search_place(query)
        except geocode.GeocodeError as e:
            st.session_state["place_options"] = []
            st.error(f"The place search service could not be reached ({e}). "
                     "You can still type coordinates such as 27.2046, 77.4977.")
        else:
            st.session_state["place_options"] = found
            st.session_state["place_params"] = (radius, days)
            if not found:
                st.warning("No place matched that. Try a larger place name, add the country, or type coordinates.")
            else:
                st.session_state["place_pick"] = 0

    options = st.session_state.get("place_options") or []
    if not options:
        st.caption("Search any town, forest, park or coordinate on Earth. TerraVision checks NASA thermal "
                   "fire detections and night-light change around it.")
        return

    idx = 0
    if len(options) > 1:
        idx = st.selectbox("Which place?", range(len(options)), key="place_pick",
                           format_func=lambda i: options[i]["name"][:110])
    place = options[idx]
    radius, days = st.session_state.get("place_params", (10, 10))
    key = (round(place["lat"], 4), round(place["lon"], 4), radius, days)

    if st.session_state.get("place_key") != key or "place_result" not in st.session_state:
        with st.spinner("Checking satellites for fire and unusual lights…"):
            st.session_state["place_result"] = placecheck.check_place(place["lat"], place["lon"], radius, days)
        st.session_state["place_key"] = key
    result = st.session_state["place_result"]

    color, icon = TONE[result["status"]]
    st.markdown(
        f'<div class="metric-card" style="--kpi:{color};margin:8px 0 14px"><div class="metric-val" style="color:{color};font-size:1.5rem">'
        f'{icon}  {html.escape(result["headline"])}</div><div class="metric-label" style="text-transform:none;letter-spacing:0;font-size:.85rem">'
        f'{html.escape(result["detail"])}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([3, 1.2])
    with left:
        st_folium(_map(place, result), width="100%", height=540, returned_objects=[],
                  key=f"place_map_{key}")
    with right:
        st.markdown("**Location**")
        st.code(f"{place['lat']:.6f}, {place['lon']:.6f}", language="text")
        st.caption(f"{format_dms(place['lat'], 'N', 'S')}  {format_dms(place['lon'], 'E', 'W')}")
        st.caption(place["name"])
        st.markdown(f"**Checked:** {result['radius_km']:g} km radius, fires from the last {result['days']} days")
        if result["hotspots"]:
            st.markdown(f"**{len(result['hotspots'])} hotspot(s)**")
            for h in result["hotspots"][:8]:
                extra = (f" · {h['when']}" if h.get("when") else "") + (f" · {h['frp']:.0f} MW" if h.get("frp") else "")
                st.caption(f"🔥 {h['distance_km']:.1f} km · {h['temp_k']:.0f} K{extra} · {h['lat']:.4f}, {h['lon']:.4f}")
        if result.get("fire_source"):
            st.caption(f"Fire data: {result['fire_source']}")
        if result["light_delta"] is not None:
            st.caption(f"Night-light change (peak within 3 km): {result['light_delta']:+.1f} nW/cm²/sr")
        for err in result["errors"]:
            st.caption(f"⚠️ {err} did not run")
        st.link_button("Open in Google Maps ↗", f"https://www.google.com/maps?q={place['lat']},{place['lon']}")
    st.caption("Satellites miss small, brief or cloud-covered fires, so a clear result means nothing was "
               "detected, not that the place is certainly safe. For emergencies call local services.")
