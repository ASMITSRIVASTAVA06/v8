"""
TerraVision AI — Weather risk section.

Shows where thunderstorms, heavy rain, damaging wind, extreme heat and tropical cyclones are
forecast to hit, flags the risky zones on the map, and can send an alert about them.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

import cyclones
import geocode
import riskzones
import weather
import weather_alerts
from alert_tools import get_smtp_config, get_telegram_config, parse_recipients
from regions import REGION_BOUNDS

IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
ATTR = "Imagery © Esri, Maxar, Earthstar Geographics"
CUSTOM = "Around a place I search…"
DETAIL = {"Coarse (about 2°)": 2.0, "Medium (about 1°)": 1.0, "Fine (about 0.5°)": 0.5}
ALERT_COLORS = {"Green": "#34d399", "Orange": "#fb923c", "Red": "#ef4444"}


@st.cache_data(ttl=1800, show_spinner=False)
def _storms():
    try:
        s, notes = cyclones.list_active_cyclones()
        return s, notes, None
    except Exception as e:
        return [], [], f"{type(e).__name__}: {str(e)[:120]}"


@st.cache_data(ttl=1800, show_spinner=False)
def _area(bounds: tuple, step: float, hours: int, storm_key: tuple):
    storms = [s for s in _storms()[0] if s["eventid"] in storm_key]
    return riskzones.assess_bounds(bounds, storms, step, hours)


def _level_chip(level: Optional[int]) -> str:
    if level is None:
        return "—"
    return f'<span class="badge" style="--bc:{weather.LEVEL_COLORS[level]}">{weather.LEVELS[level]}</span>'


def _map(bounds, area: Optional[Dict], storms: List[Dict], min_level: int) -> folium.Map:
    min_lon, min_lat, max_lon, max_lat = bounds
    m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=6, tiles=None,
                   control_scale=True, prefer_canvas=True)
    folium.TileLayer(IMAGERY, attr=ATTR, name="Satellite imagery", max_zoom=19).add_to(m)
    folium.TileLayer(LABELS, attr=ATTR, name="Place labels", overlay=True, max_zoom=19).add_to(m)
    Fullscreen().add_to(m)
    folium.Rectangle([[min_lat, min_lon], [max_lat, max_lon]], color="#7db4ff", weight=1, dash_array="6 6",
                     fill=False, tooltip="Area checked").add_to(m)

    cells = folium.FeatureGroup(name="Risk zones")
    for c in (area or {}).get("cells", []):
        col = weather.LEVEL_COLORS[c["level"]]
        shown = c["level"] >= max(1, min_level)
        folium.Rectangle(
            [[c["lat"] - c["half_lat"], c["lon"] - c["half_lon"]], [c["lat"] + c["half_lat"], c["lon"] + c["half_lon"]]],
            color=col, weight=1.5 if shown else 0.5, fill=shown, fill_color=col, fill_opacity=0.28 if shown else 0,
            tooltip=f"{weather.LEVELS[c['level']]} · {riskzones.zone_label(c)} · {c['headline']}",
            popup=folium.Popup("<br>".join(
                [f"<b>{weather.LEVELS[c['level']]}</b> · {riskzones.zone_label(c)}"] +
                [f"{weather.HAZARD_LABELS[k]}: <b>{h['value']} {h['unit']}</b> ({weather.LEVELS[h['level']]})"
                 + (f" · {h['at'].replace('T', ' ')} UTC" if h.get("at") else "")
                 for k, h in c["hazards"].items() if h["level"] > 0]), max_width=300),
        ).add_to(cells)
    cells.add_to(m)

    tracks = folium.FeatureGroup(name="Cyclone tracks")
    for s in storms:
        col = ALERT_COLORS.get(s["alert"], "#fb923c")
        obs = [(p["lat"], p["lon"]) for p in s["track"] if p["actual"]]
        fc = [(p["lat"], p["lon"]) for p in s["track"] if not p["actual"]]
        if len(obs) > 1:
            folium.PolyLine(obs, color=col, weight=3, tooltip=f"{s['name']} observed track").add_to(tracks)
        if fc:
            folium.PolyLine(([obs[-1]] if obs else []) + fc, color=col, weight=3, dash_array="8 8",
                            tooltip=f"{s['name']} forecast track").add_to(tracks)
        for p in s["track"]:
            if p.get("radius_km") and not p["actual"]:
                folium.Circle([p["lat"], p["lon"]], radius=p["radius_km"] * 1000, color=col, weight=1,
                              fill=True, fill_opacity=0.08, tooltip="Forecast gale-force (34 kt) wind field").add_to(tracks)
        folium.Marker([s["lat"], s["lon"]], tooltip=f"{s['name']} · {s['alert']} alert · {s['severity_text']}",
                      icon=folium.Icon(color="red" if s["alert"] == "Red" else "orange", icon="cloud", prefix="fa")).add_to(m)
    tracks.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def _area_selector():
    c1, c2, c3, c4 = st.columns([3, 1.4, 1.6, 1.6])
    choice = c1.selectbox("Area", list(REGION_BOUNDS) + [CUSTOM], key="wx_area")
    hours = c2.selectbox("Forecast", [24, 48, 72, 120], index=2, format_func=lambda h: f"next {h} h", key="wx_hours")
    detail = c3.selectbox("Detail", list(DETAIL), index=1, key="wx_detail")
    min_level = c4.selectbox("Flag from", [1, 2, 3], index=0, format_func=lambda i: weather.LEVELS[i], key="wx_min")
    bounds, name = None, choice
    if choice == CUSTOM:
        d1, d2 = st.columns([4, 1.4])
        q = d1.text_input("Place or coordinates", placeholder="e.g. Bhubaneswar  ·  20.3, 85.8", key="wx_q")
        radius = d2.selectbox("Radius", [100, 200, 400], format_func=lambda k: f"{k} km", key="wx_r")
        if q.strip():
            try:
                found = geocode.search_place(q, limit=1)
            except geocode.GeocodeError as e:
                st.error(f"Place search unavailable ({e}). Type coordinates such as 20.3, 85.8.")
                found = []
            if found:
                bounds = weather.bounds_around(found[0]["lat"], found[0]["lon"], radius)
                name = found[0]["name"][:60]
            elif q.strip():
                st.warning("No place matched that.")
    else:
        bounds = REGION_BOUNDS[choice]
    return name, bounds, hours, DETAIL[detail], min_level


def render_weather_page() -> None:
    storms, notes, storm_err = _storms()
    tab_map, tab_regions, tab_cyc, tab_alert = st.tabs(
        ["🗺️  Risk map", "🌲  All forest regions", "🌀  Cyclones", "📣  Send alert"])

    with tab_map:
        name, bounds, hours, step, min_level = _area_selector()
        if bounds is None:
            st.caption("Pick a forest region, or search a place, to see forecast thunderstorms, heavy rain, wind, heat "
                       "and cyclone tracks on the map.")
        else:
            with st.spinner("Reading the forecast…"):
                try:
                    area = _area(tuple(bounds), step, hours, tuple(s["eventid"] for s in storms))
                except Exception as e:
                    area = None
                    st.error(f"Could not get the forecast ({type(e).__name__}: {str(e)[:100]}).")
            if area is not None and not area["cells"]:
                st.error("The forecast service returned nothing for this area, so no risk can be stated. "
                         + (area["errors"][0] if area["errors"] else ""))
            elif area is not None:
                worst = area["worst"]
                col = weather.LEVEL_COLORS[worst]
                top = max(area["cells"], key=lambda c: c["level"])
                st.markdown(
                    f'<div class="metric-card" style="--kpi:{col};margin:6px 0 14px"><div class="metric-val" style="color:{col};font-size:1.5rem">'
                    f'{"🟢🟡🟠🔴"[worst]}  {html.escape(top["headline"] if worst else "No significant weather hazard forecast")}</div>'
                    f'<div class="metric-label" style="text-transform:none;letter-spacing:0;font-size:.85rem">'
                    f'{html.escape(name)} · next {hours} h · {sum(c["level"] >= min_level for c in area["cells"])} of '
                    f'{len(area["cells"])} zones flagged · forecast {area["generated_at"]}</div></div>', unsafe_allow_html=True)
                if area["missing"]:
                    st.warning(f"{area['missing']} zone(s) could not be forecast and are not shown. "
                               "Missing zones are unknown, not safe.")
                st_folium(_map(bounds, area, storms, min_level), width="100%", height=560, returned_objects=[],
                          key=f"wx_map_{name}_{hours}_{step}_{min_level}")
                fl = weather.flagged(area["cells"], min_level)
                st.markdown('<div class="section-title">Flagged zones</div>', unsafe_allow_html=True)
                if not fl:
                    st.success("No zone is flagged at this level in the chosen window.")
                else:
                    df = pd.DataFrame([{
                        "Level": weather.LEVELS[c["level"]], "Zone (centre)": riskzones.zone_label(c),
                        "Hazard": c["headline"].split(": ", 1)[-1],
                        "Peak rain (mm/24 h)": c["hazards"]["rain"]["value"], "Peak gust (km/h)": c["hazards"]["wind"]["value"],
                        "Storm energy (CAPE)": c["hazards"]["storm"]["value"],
                        "Peak time (UTC)": next((h["at"].replace("T", " ") for h in c["hazards"].values()
                                                 if h["level"] == c["level"] and h.get("at")), ""),
                        "Lat": c["lat"], "Lon": c["lon"]} for c in fl])
                    st.dataframe(df, hide_index=True)
                    st.download_button("⬇️  Download flagged zones (CSV)", df.to_csv(index=False).encode("utf-8"),
                                       "terravision_weather_zones.csv", "text/csv")
        _footer()

    with tab_regions:
        st.caption("Coarse check of every forest region you monitor. One forecast request per few zones, cached for 30 minutes.")
        hrs = st.selectbox("Forecast window", [24, 48, 72, 120], index=2, format_func=lambda h: f"next {h} h", key="wx_ov_h")
        if st.button("Check all forest regions", key="wx_ov_go"):
            with st.spinner("Checking every region…"):
                st.session_state["wx_overview"] = riskzones.region_overview(storms, hours=hrs)
        rows = st.session_state.get("wx_overview")
        if rows:
            ov = pd.DataFrame([{"Region": r["region"], "Level": "Could not check" if r["level"] is None else weather.LEVELS[r["level"]],
                                "What": r["headline"], "Zones at Warning+": f"{r['flagged']} of {r['cells']}",
                                "Peak (UTC)": (r.get("peak_at") or "").replace("T", " ")} for r in rows])
            st.dataframe(ov, hide_index=True)
            bad = [r["region"] for r in rows if r["level"] is None]
            if bad:
                st.warning("Could not check: " + ", ".join(bad) + ". These are unknown, not safe.")
        _footer()

    with tab_cyc:
        if storm_err:
            st.error(f"The cyclone feed could not be reached ({storm_err}). No cyclone information is shown, "
                     "which does not mean there are none.")
        elif not storms:
            st.success("GDACS lists no active tropical cyclone right now.")
        for n in notes:
            st.caption(f"ℹ️ {n}")
        for s in storms:
            col = ALERT_COLORS.get(s["alert"], "#fb923c")
            st.markdown(f'<div class="metric-card" style="--kpi:{col};margin:8px 0"><div class="metric-val" style="color:{col};font-size:1.3rem">'
                        f'🌀 {html.escape(s["name"])}</div><div class="metric-label" style="text-transform:none;letter-spacing:0;font-size:.85rem">'
                        f'{s["alert"]} alert · {html.escape(s["severity_text"] or "tropical cyclone")} · at {s["lat"]:.1f}°, {s["lon"]:.1f}° · '
                        f'{len(s["track"])} track point(s) · source {html.escape(s["source"] or "GDACS")}</div></div>', unsafe_allow_html=True)
        if storms:
            lat = sum(s["lat"] for s in storms) / len(storms)
            lon = sum(s["lon"] for s in storms) / len(storms)
            st_folium(_map((lon - 25, lat - 15, lon + 25, lat + 15), None, storms, 1), width="100%", height=480,
                      returned_objects=[], key="wx_cyc_map")
            st.markdown('<div class="section-title">Which forest regions are exposed</div>', unsafe_allow_html=True)
            exp = []
            for r, b in REGION_BOUNDS.items():
                t = cyclones.worst_cyclone_threat((b[1] + b[3]) / 2, (b[0] + b[2]) / 2, storms)
                if t and t["level"] > 0:
                    exp.append({"Region": r, "Storm": t["storm"], "Level": weather.LEVELS[t["level"]],
                                "Closest approach (km)": t["distance_km"], "Why": t["reason"]})
            st.dataframe(pd.DataFrame(exp), hide_index=True) if exp else st.caption("No monitored forest region is near an active cyclone.")
        _footer()

    with tab_alert:
        st.caption("Send the current weather and cyclone watch for your forest regions to your team. "
                   "Uses the same e-mail and Telegram settings as the forest alerts.")
        c1, c2 = st.columns([3, 1])
        to_text = c1.text_input("E-mail recipients (comma-separated)", key="wx_to")
        level = c2.selectbox("Alert from", [1, 2, 3], index=1, format_func=lambda i: weather.LEVELS[i], key="wx_alert_lvl")
        use_tg = st.checkbox("Also send to Telegram", key="wx_tg")
        rows = st.session_state.get("wx_overview")
        if not rows:
            st.info("Run **Check all forest regions** first, so the alert reflects the latest forecast.")
        else:
            msg = weather_alerts.build_message(rows, storms, level)
            if msg is None:
                st.success(f"No region is at {weather.LEVELS[level]} or above, so there is nothing to send.")
            else:
                st.text_area("Message preview", msg["body"], height=220, disabled=True)
                if st.button("📣  Send weather alert now", key="wx_send"):
                    recipients, invalid = parse_recipients(to_text)
                    if invalid:
                        st.warning(f"Ignoring invalid addresses: {', '.join(invalid)}")
                    if not (recipients or use_tg):
                        st.error("Add at least one valid e-mail address or tick Telegram.")
                    else:
                        for line in weather_alerts.deliver(msg, recipients, use_tg):
                            (st.error if "FAILED" in line else st.warning if "skipped" in line else st.success)(line)
        _footer()


def _footer() -> None:
    st.caption("Forecasts come from Open-Meteo model data and cyclone advisories from GDACS (NOAA / JTWC). This is an "
               "indication, not an official warning: follow IMD, NHC, JTWC or your national weather service.")
