import html

import streamlit as st
import folium
from streamlit_folium import st_folium

import store
import theme
from alert_tools import CATEGORY_STYLE, category_label
from alerts_view import render_alerts_page
from dashboard import render_dashboard
from pipeline import run_scan
from evidence import zone_of
from regions import REGION_BOUNDS, REGION_COORDS

# Page configuration for the TerraVision AI dashboard
st.set_page_config(
    page_title="TerraVision AI — Forest Guardian",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded",
)

# Satellite data strategy (see pipeline.py): the newest available VIIRS months are
# downloaded from Earth Engine and cached in ./data_cache; previously exported
# local .tif files are only a fallback when Earth Engine cannot be reached.

# One-time import of scans saved by the earlier JSON-file version
if "legacy_imported" not in st.session_state:
    try:
        store.import_legacy_json()
    except Exception:
        pass
    st.session_state["legacy_imported"] = True


# Store the current theme in session state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Theme toggle button in the top-right corner
col_gap, col_toggle = st.columns([10, 1])
with col_toggle:
    toggle_label = "☀️" if st.session_state.dark_mode else "🌙"
    if st.button(toggle_label, help="Toggle light/dark mode", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

is_dark = st.session_state.dark_mode

theme.apply(st, is_dark)

# Colour and label for each alert category now live in alert_tools.CATEGORY_STYLE
# so the Dashboard and Alerts sections use exactly the same palette.

# Sidebar
with st.sidebar:
    st.markdown(theme.brand_html(), unsafe_allow_html=True)
    st.markdown('<div class="section-title">Navigate</div>', unsafe_allow_html=True)
    try:
        _n_alerts = store.alert_count()
    except Exception:
        _n_alerts = 0
    _nav_labels = {
        "monitor":   "◎  Monitor",
        "dashboard": "▦  Dashboard",
        "alerts":    f"◈  Alerts · {_n_alerts}" if _n_alerts else "◈  Alerts",
        "search":    "⌕  Search a place",
        "weather":   "☈  Weather risk",
    }
    page = st.radio("Section", list(_nav_labels), format_func=_nav_labels.get,
                    label_visibility="collapsed", key="nav_page")

    st.markdown('<div class="section-title">Forest Region</div>', unsafe_allow_html=True)

    # All regions in a single flat list for the selectbox
    region = st.selectbox("Select region", list(REGION_COORDS.keys()),
                          label_visibility="collapsed")

    st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)

    # Minimum radiance delta to flag a pixel as suspicious
    threshold = st.slider(
        "Alert threshold (nW/cm²/sr)",
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="Higher = fewer but more certain alerts. Recommended: 5–8 to reduce noise."
    )

    # Minimum pixels in a cluster — filters out single-pixel sensor noise
    min_cluster = st.slider(
        "Min cluster size (pixels)",
        min_value=1, max_value=20, value=3,
        help="Filter out lone pixels. Camps typically appear as 2–8 pixel clusters."
    )

    regional_fix = st.checkbox(
        "Correct for regional brightening", value=True,
        help="Whole districts get brighter every year as villages are electrified. This removes that "
             "background trend so only patches that stand out from their surroundings are flagged. "
             "Turn off to reproduce older results.")

    st.markdown('<div class="section-title">Data & Context</div>', unsafe_allow_html=True)
    use_context = st.checkbox(
        "Cross-check with other satellites", value=True,
        help="Adds protected-area, land-cover, vegetation, forest-loss and active-fire evidence to every "
             "alert and adjusts its confidence. Slower (about a minute extra), needs Earth Engine.")
    hide_urban = st.checkbox("Hide urban lights", value=True,
                             help="Hides alerts where built-up land covers 30 % or more of the pixel.")
    protected_only = st.checkbox("Forest & protected areas only", value=True,
                                 help="Hides alerts outside the region's real boundary: an official boundary file if you "
                                      "added one, otherwise designated protected areas or land surrounded by forest.")
    refresh_data = st.checkbox("Force fresh satellite download", value=False,
                               help="Ignore cached rasters and download the newest data again.")

    try:
        import modelcard
        st.caption(modelcard.summary(modelcard.read()))
    except Exception:
        pass

    try:
        import apis
        with st.expander("Data sources"):
            _icon = {"live": "🟢", "free": "🟡", "missing": "🔴"}
            for _r in apis.status_rows():
                st.markdown(f"{_icon[_r['state']]} **{_r['name']}** · {_r['detail']}")
            st.caption("🟢 key set · 🟡 free service, no key · 🔴 key missing. Put keys in `.env` (see README).")
    except Exception:
        pass

    st.markdown('<div class="section-title">Filter by Type</div>', unsafe_allow_html=True)
    show_encroachment = st.checkbox("Illegal encroachment", value=True)
    show_wildfire     = st.checkbox("Natural wildfires",    value=True)
    show_agri         = st.checkbox("Agricultural burns",   value=True)
    show_anomaly      = st.checkbox("Unclassified anomaly", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color: var(--faint); line-height: 1.8;'>
    <b style='color:var(--muted)'>Data</b> NASA VIIRS DNB · 500 m<br>
    <b style='color:var(--muted)'>Baseline</b> 5-yr median 2019–2023<br>
    <b style='color:var(--muted)'>Cross-checks</b> WDPA · WorldCover · Hansen · FIRMS · Sentinel-2<br><br>
    Research and awareness prototype. Verify alerts on the ground before acting.
    </div>
    """, unsafe_allow_html=True)

# Header
try:
    import modelcard
    _card = modelcard.read() or {}
    _clf = "real-data model" if _card.get("provenance") == "real" else "synthetic model"
except Exception:
    _clf = "synthetic model"
_titles = {"monitor": "Monitor", "dashboard": "Dashboard", "alerts": "Alerts", "search": "Search", "weather": "Weather risk"}
_subs = {
    "monitor": "Scan a forest region, see every night-light cluster on satellite imagery, and judge it with independent evidence.",
    "dashboard": "Patterns across your scans: what is changing, where, and how much to trust it.",
    "alerts": "Review, triage and share every alert. Your verdicts make the classifier better.",
    "search": "Look up any place on Earth and see whether satellites detect fire or unusual lights around it.",
    "weather": "Forecast thunderstorms, heavy rain, wind, heat and tropical cyclones, and flag the zones they may hit.",
}
st.markdown(theme.hero_html(
    f"TerraVision · {_titles.get(page, 'Monitor')}", _subs.get(page, ""),
    [("Region", region), ("Classifier", _clf), ("Resolution", "500 m")]), unsafe_allow_html=True)

def render_monitor():
    """Original map + satellite scan view."""
    # Legend
    st.markdown(theme.legend_html([
        (CATEGORY_STYLE["illegal_encroachment"]["color"], "Illegal encroachment: small persistent cluster, moderate radiance"),
        (CATEGORY_STYLE["natural_wildfire"]["color"], "Natural wildfire: large cluster, very high radiance spike"),
        (CATEGORY_STYLE["agricultural_burn"]["color"], "Agricultural burn: large seasonal cluster, moderate radiance"),
        (CATEGORY_STYLE["anomaly"]["color"], "Unclassified anomaly: does not match known patterns"),
    ]), unsafe_allow_html=True)

    # Scan button
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        scan = st.button("🔍  Run Satellite Scan", use_container_width=True)

    # Map setup
    center = REGION_COORDS.get(region, [12.0, 76.5])
    if is_dark:
        tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    else:
        tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"

    m = folium.Map(location=center, zoom_start=6, tiles=None, prefer_canvas=True)
    folium.TileLayer(
        tiles=tile_url,
        attr="Tiles © Esri — Esri, HERE, Garmin, OpenStreetMap contributors",
        max_zoom=16, name="Map",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Imagery © Esri, Maxar, Earthstar Geographics",
        max_zoom=19, name="Satellite imagery", show=False,
    ).add_to(m)

    min_lon, min_lat, max_lon, max_lat = REGION_BOUNDS.get(
        region, (center[1] - 3, center[0] - 3, center[1] + 3, center[0] + 3)
    )
    box_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    folium.Rectangle(
        bounds=box_bounds,
        color="#7db4ff", weight=1, dash_array="6 6", fill=False,
        tooltip="Search box: the area the satellite data is downloaded for (not the forest itself)",
    ).add_to(m)
    try:
        import boundaries
        bnd = boundaries.load_boundary(region)
        gj = boundaries.map_geojson(bnd)
        if gj:
            folium.GeoJson(
                gj, name="Forest / protected boundary",
                style_function=lambda f: {"color": "#22d3ee", "weight": 1.6,
                                          "fillColor": "#22d3ee", "fillOpacity": 0.14},
                tooltip=folium.GeoJsonTooltip(fields=["NAME"], aliases=["Area"], labels=True)
                if bnd["features"] and "NAME" in (bnd["features"][0].get("properties") or {}) else None,
            ).add_to(m)
            st.caption("Region boundary: " + boundaries.describe(bnd))
        else:
            st.caption("Region boundary: " + boundaries.describe(None))
    except Exception:
        pass
    m.fit_bounds(box_bounds)

    # Run a scan (real data) and keep it; the map always shows the newest scan of this region
    if scan:
        with st.status("🛰️  Scanning — this can take a few minutes the first time…", expanded=True) as status:
            try:
                result = run_scan(
                    region, threshold=threshold, min_cluster=min_cluster, enrich=use_context,
                    refresh=refresh_data, regional_correction=regional_fix, progress=lambda msg: status.update(label=f"🛰️  {msg}"),
                )
                st.session_state["monitor_scan"] = result
                status.update(label=f"✅  Scan complete — {len(result['alerts'])} alerts", state="complete",
                              expanded=False)
            except Exception as e:
                status.update(label="Scan failed", state="error")
                st.error(f"Could not complete the scan: {e}")

    view = st.session_state.get("monitor_scan")
    if not view or view["region"] != region:
        saved = [s for s in store.latest_scans() if s["region"] == region]
        view = saved[0] if saved else None

    if view:
        all_alerts = view["alerts"]
        ctx = set(view.get("context_ok", []))
        active_cats = set()
        if show_encroachment: active_cats.add("illegal_encroachment")
        if show_wildfire:     active_cats.add("natural_wildfire")
        if show_agri:         active_cats.add("agricultural_burn")
        if show_anomaly:      active_cats.add("anomaly")

        by_type = [a for a in all_alerts if a["category"] in active_cats]
        alerts = by_type
        n_urban = n_outside = 0
        if hide_urban and "land_cover" in ctx:
            n_urban = sum(1 for a in alerts if a.get("urban_light"))
            alerts = [a for a in alerts if not a.get("urban_light")]
        if protected_only and "protected_area" in ctx:
            n_outside = sum(1 for a in alerts if zone_of(a) == "outside")
            alerts = [a for a in alerts if zone_of(a) != "outside"]

        st.caption(f"Scan of **{view['region']}** · satellite period **{view.get('period') or 'unknown'}** · "
                   f"run {view['scanned_at']}")
        for note in view.get("notes", []):
            st.info(note)
        if use_context and not ctx and all_alerts:
            st.info("This scan has no cross-check evidence (it was run without cross-checks or Earth Engine "
                    "was unavailable), so the urban and protected-area filters have nothing to work with.")

        n_enc  = sum(1 for a in alerts if a["category"] == "illegal_encroachment")
        n_fire = sum(1 for a in alerts if a["category"] == "natural_wildfire")
        n_agri = sum(1 for a in alerts if a["category"] == "agricultural_burn")
        avg_conf = int(sum(a["confidence"] for a in alerts) / len(alerts)) if alerts else 0
        n_new = sum(1 for a in alerts if a.get("lifecycle") == "new")
        n_rec = sum(1 for a in alerts if a.get("lifecycle") == "recurring")
        n_res = len(view.get("resolved", []))

        _c = CATEGORY_STYLE
        st.markdown(theme.kpi_html([
            (n_enc, "Illegal encroachments", _c["illegal_encroachment"]["color"]),
            (n_fire, "Natural wildfires", _c["natural_wildfire"]["color"]),
            (n_agri, "Agricultural burns", _c["agricultural_burn"]["color"]),
            (f"{avg_conf}%", "Avg confidence", theme.palette(is_dark)["primary2"]),
        ]) + theme.kpi_html([
            (n_new, "New this scan", theme.palette(is_dark)["warn"]),
            (n_rec, "Recurring", theme.palette(is_dark)["primary2"]),
            (n_res, "Resolved since last scan", theme.palette(is_dark)["good"]),
            (n_urban + n_outside, "Hidden (urban / outside forest)", theme.palette(is_dark)["faint"]),
        ]), unsafe_allow_html=True)

        _strong = sum(1 for a in alerts if a.get("verdict") == "Strong evidence")
        _pal = theme.palette(is_dark)
        g1, g2 = st.columns(2)
        g1.markdown(theme.gauge_html(avg_conf, "Average confidence",
                                     "Evidence-adjusted: model score plus what other satellites confirm.",
                                     _pal["accent"]), unsafe_allow_html=True)
        g2.markdown(theme.gauge_html(100 * _strong / len(alerts) if alerts else 0, "Strongly corroborated",
                                     "Share of alerts backed by two or more independent signals.",
                                     _pal["good"]), unsafe_allow_html=True)

        if n_enc > 0:
            st.warning(f"🚨  {n_enc} illegal encroachment cluster(s) detected. Immediate verification recommended.")
        elif all_alerts:
            st.success("✅  No illegal encroachments left after filtering.")
        else:
            st.success("✅  No illegal encroachments detected in this scan.")

        for alert in alerts:
            style   = CATEGORY_STYLE.get(alert["category"], CATEGORY_STYLE["anomaly"])
            color   = style["color"]
            radius  = 10 if alert["category"] == "illegal_encroachment" else 7
            strong  = alert.get("verdict") == "Strong evidence"
            doubtful = alert.get("verdict") == "Likely false positive"
            opacity = 0.35 if doubtful else (0.9 if alert["severity"] in ("high", "critical") else 0.65)
            evidence_html = "".join(f"<br>• {html.escape(str(e))}" for e in alert.get("evidence", [])[:5])

            popup_html = f"""
            <div style='font-family:sans-serif;font-size:13px;min-width:230px;'>
                <b style='color:{color}'>{style["label"].upper()}</b> · {html.escape(str(alert.get('alert_id', '')))}<br>
                <hr style='margin:6px 0;border-color:#333'>
                📍 {alert['lat']:.5f}, {alert['lon']:.5f}<br>
                ⚡ Radiance mean: <b>{alert['delta_mean']} nW/cm²/sr</b><br>
                ⚡ Radiance peak: <b>{alert['delta_max']} nW/cm²/sr</b><br>
                🔲 Cluster size: {alert['cluster_size']} pixels<br>
                🎯 Confidence: <b>{alert['confidence']}%</b> (model {alert.get('model_confidence', alert['confidence'])}%)<br>
                ⚠️ Severity: {alert['severity'].upper()}<br>
                🔁 {alert.get('lifecycle', 'new').capitalize()} · seen {alert.get('times_seen', 1)}×<br>
                🧭 <b>{html.escape(str(alert.get('verdict', '')))}</b>{evidence_html}
            </div>
            """
            folium.CircleMarker(
                location=[alert["lat"], alert["lon"]],
                radius=radius, color="#ffffff" if strong else color, fill=True,
                fill_color=color, fill_opacity=opacity, weight=2.5 if strong else 1.5,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{style['label']} · {alert['confidence']}% · {alert.get('verdict', '')}",
            ).add_to(m)

        enc_alerts = [a for a in alerts if a["category"] == "illegal_encroachment"]
        if enc_alerts:
            with st.expander(f"📋  Illegal encroachment detail log ({len(enc_alerts)} clusters)"):
                rows = "\n".join([
                    f"| {a.get('alert_id', i + 1)} | {a['lat']:.5f}, {a['lon']:.5f} "
                    f"| {a['delta_mean']} nW | {a['cluster_size']} px | {a['confidence']}% "
                    f"| {a.get('verdict', '')} | {a.get('lifecycle', '')} |"
                    for i, a in enumerate(enc_alerts[:50])
                ])
                st.markdown(
                    "| ID | Coordinates | Δ Mean | Cluster | Conf | Evidence | Lifecycle |\n"
                    "|----|-------------|--------|---------|------|----------|-----------|\n" + rows
                )
    else:
        st.info("No scan for this region yet. Press **Run Satellite Scan** to analyse the latest "
                "satellite data; results are saved and appear on the Dashboard and Alerts sections.")

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, width="100%", height=540, returned_objects=[])

    with st.expander("🩺  Scan history & health"):
        try:
            runs = store.recent_runs()
        except Exception as e:
            runs = None
            st.warning(f"Could not read the scan history: {e}")
        if runs is None:
            pass
        elif runs.empty:
            st.caption("No scans recorded yet.")
        else:
            n_failed = int((runs["status"] == "failed").sum())
            st.caption(f"Last {len(runs)} scan attempts — {n_failed} failed. "
                       "Details are also written to logs/terravision.log.")
            st.dataframe(runs, hide_index=True)


def _alert_source():
    """
    Real scan results only, straight from the database: the newest scan of each region.
    Returns a dict: label, alerts, resolved, notes, context_ok.
    """
    scans = store.latest_scans()
    if not scans:
        return {"label": "No scans yet", "alerts": [], "resolved": [], "notes": [], "context_ok": []}

    options = {}
    if len(scans) > 1:
        options[f"All scanned regions ({len(scans)})"] = {
            "alerts": [a for s in scans for a in s["alerts"]],
            "resolved": [r for s in scans for r in s.get("resolved", [])],
            "notes": [n for s in scans for n in s.get("notes", [])],
            "context_ok": sorted(set.intersection(*[set(s.get("context_ok", [])) for s in scans])),
        }
    for s in scans:
        options[f"{s['region']} — scanned {s['scanned_at']} (period {s.get('period') or 'unknown'})"] = s
    choice = st.selectbox("Data source (real VIIRS scan results)", list(options), key="data_source")
    pick = options[choice]
    return {"label": choice, "alerts": pick["alerts"], "resolved": pick.get("resolved", []),
            "notes": pick.get("notes", []), "context_ok": pick.get("context_ok", [])}


if page == "monitor":
    render_monitor()
elif page == "dashboard":
    _src = _alert_source()
    render_dashboard(_src, is_dark, history=store.scan_history())
elif page == "weather":
    from weather_view import render_weather_page
    render_weather_page()
elif page == "search":
    from search_view import render_search_page
    render_search_page()
else:
    _src = _alert_source()
    render_alerts_page(_src)

st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:var(--footer-color); padding: 8px 0;'>
    TerraVision AI · Built with NASA VIIRS · Protecting forests from space 🛰️
</div>
""", unsafe_allow_html=True)