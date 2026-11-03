"""
TerraVision AI — active tropical cyclones.

Reads the Global Disaster Alert and Coordination System (GDACS, a UN/EU service) list of tropical
cyclones, which republishes advisories from NOAA and the Joint Typhoon Warning Center, together with
each storm's observed and forecast track and wind radii. For any point we then work out how close the
forecast track and wind field come.

This is an indication built on public advisories, not an official warning. For India follow IMD
(mausam.imd.gov.in); for the Atlantic and East Pacific, NHC; elsewhere your national service / JTWC.

The parsing follows GDACS's public API as documented by third-party clients; it is written to
tolerate missing fields, but if GDACS changes its format `list_active_cyclones` returns an error
note instead of guessing.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from reliability import call_with_retry, get_logger
from weather import haversine_km  # noqa: F401  (re-exported for callers)

BASE = "https://www.gdacs.org/gdacsapi/api"
NM_TO_KM = 1.852
ALERT_LEVELS = ["Green", "Orange", "Red"]
HURRICANE_KT = 64.0
STORM_KT = 34.0

# distance bands (km) from the nearest forecast track point
WARNING_KM = 300.0
WATCH_KM = 600.0


class CycloneError(RuntimeError):
    """GDACS could not be reached or returned something unusable."""


def _http_json(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers={"User-Agent": "TerraVisionAI/1.0 (forest monitoring research tool)",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


# ── list of active storms ────────────────────────────────────────────────────

def parse_event_list(data: Dict) -> List[Dict]:
    """Storms from a GDACS event-list GeoJSON: [{eventid, name, alert, lat, lon, current, from, to, wind_kt, text}]."""
    out = []
    for feat in (data or {}).get("features", []) or []:
        p = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2 or p.get("eventid") is None:
            continue
        sev = p.get("severitydata") or {}
        out.append({
            "eventid": int(p["eventid"]), "name": p.get("eventname") or p.get("name") or f"TC {p['eventid']}",
            "alert": p.get("alertlevel") or "Green", "lon": _f(coords[0]), "lat": _f(coords[1]),
            "current": _truthy(p.get("iscurrent")), "from": p.get("fromdate"), "to": p.get("todate"),
            "wind_kmh": _f(sev.get("severity")), "severity_text": sev.get("severitytext") or "",
            "source": p.get("source") or "", "track": [],
        })
    return [s for s in out if s["lat"] is not None and s["lon"] is not None]


def _find_key(obj, key: str):
    """First value stored under `key` anywhere inside nested dicts/lists, else None."""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], str):
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r:
                return r
    return None


def parse_timeline(data: Dict) -> List[Dict]:
    """Track points from a GDACS timeline: [{lat, lon, actual, wind_kt, gust_kt, pressure, radius_km, time}]."""
    items = ((data or {}).get("channel") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    pts = []
    for it in items:
        lat, lon = _f(it.get("latitude")), _f(it.get("longitude"))
        if lat is None or lon is None:
            continue
        radii = [_f(it.get(f"windrad_nm_34kt_{q}")) for q in ("ne", "se", "sw", "nw")]
        radii = [r for r in radii if r]
        pts.append({
            "lat": lat, "lon": lon, "actual": _truthy(it.get("actual")),
            "wind_kt": _f(it.get("wind_speed")), "gust_kt": _f(it.get("wind_gusts")), "pressure": _f(it.get("pressure")),
            "radius_km": round(max(radii) * NM_TO_KM, 1) if radii else None,
            "time": it.get("datetime") or it.get("forecast_datetime") or it.get("advisory_datetime") or "",
            "status": (it.get("storm_status") or [""])[0] if isinstance(it.get("storm_status"), list) else it.get("storm_status") or "",
        })
    return pts


def list_active_cyclones(http: Optional[Callable[[str], object]] = None, today: Optional[date] = None,
                         with_tracks: bool = True) -> Tuple[List[Dict], List[str]]:
    """
    Active cyclones worldwide, with forecast tracks when available: (storms, notes).
    Raises CycloneError only when the list itself cannot be fetched.
    """
    http = http or _http_json
    today = today or date.today()
    notes: List[str] = []
    by_id: Dict[int, Dict] = {}
    failures = 0
    for level in ALERT_LEVELS:
        q = urllib.parse.urlencode({"eventlist": "TC", "alertlevel": level, "pageSize": 100, "pageNumber": 1,
                                    "fromDate": (today - timedelta(days=10)).isoformat(), "toDate": today.isoformat()})
        try:
            data = call_with_retry(http, f"{BASE}/events/geteventlist/SEARCH?{q}")
        except Exception as e:
            failures += 1
            notes.append(f"GDACS {level} list: {type(e).__name__}")
            continue
        for s in parse_event_list(data):
            if s["current"]:
                by_id[s["eventid"]] = s
    if failures == len(ALERT_LEVELS):
        raise CycloneError("; ".join(notes) or "GDACS unreachable")

    storms = sorted(by_id.values(), key=lambda s: (ALERT_LEVELS.index(s["alert"]) if s["alert"] in ALERT_LEVELS else 0),
                    reverse=True)
    if with_tracks:
        for s in storms:
            try:
                ev = call_with_retry(http, f"{BASE}/events/geteventdata?eventtype=TC&eventid={s['eventid']}")
                url = _find_key(ev, "timeline")
                if url:
                    s["track"] = parse_timeline(call_with_retry(http, url))
                if not s["track"]:
                    notes.append(f"{s['name']}: no forecast track published; using its current position only")
            except Exception as e:
                get_logger().warning("Cyclone track failed for %s: %s", s["name"], e)
                notes.append(f"{s['name']}: track unavailable ({type(e).__name__})")
    for s in storms:                         # a storm always has at least its current position
        if not s["track"]:
            s["track"] = [{"lat": s["lat"], "lon": s["lon"], "actual": True, "wind_kt": (s["wind_kmh"] or 0) / 1.852 or None,
                           "gust_kt": None, "pressure": None, "radius_km": None, "time": "", "status": ""}]
    return storms, notes


# ── distance and risk ────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2)


def assess_cyclone(lat: float, lon: float, storm: Dict) -> Dict:
    """
    How close `storm`'s observed + forecast track and 34-knot wind field come to (lat, lon).
    Returns {level 0-3, distance_km, nearest (track point), in_wind_field, reason}.
    """
    best, best_d = None, float("inf")
    in_field = False
    for p in storm["track"]:
        d = haversine(lat, lon, p["lat"], p["lon"])
        if d < best_d:
            best, best_d = p, d
        if p.get("radius_km") and d <= p["radius_km"]:
            in_field = True
    peak_kt = max([p.get("wind_kt") or 0 for p in storm["track"]] + [0])
    hurricane = peak_kt >= HURRICANE_KT
    if in_field:
        level, why = (3 if hurricane else 2), "inside the forecast gale-force (34 kt) wind field"
    elif best_d <= WARNING_KM:
        level, why = (2 if hurricane or peak_kt >= STORM_KT else 1), f"forecast track passes within {WARNING_KM:.0f} km"
    elif best_d <= WATCH_KM:
        level, why = 1, f"forecast track passes within {WATCH_KM:.0f} km"
    else:
        level, why = 0, "well away from the forecast track"
    return {"level": level, "distance_km": round(best_d, 1), "nearest": best, "in_wind_field": in_field,
            "peak_wind_kt": round(peak_kt) if peak_kt else None, "reason": why, "storm": storm["name"]}


def worst_cyclone_threat(lat: float, lon: float, storms: Iterable[Dict]) -> Optional[Dict]:
    """The most serious threat to the point from any storm (ties broken by distance), or None."""
    threats = [assess_cyclone(lat, lon, s) for s in storms]
    if not threats:
        return None
    return max(threats, key=lambda t: (t["level"], -t["distance_km"]))
