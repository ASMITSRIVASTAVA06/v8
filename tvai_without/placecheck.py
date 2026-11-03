"""
TerraVision AI — "is anything burning here?" check for any place on Earth.

Looks at two independent satellite signals around a point:
  1. Active fire: NASA FIRMS thermal hotspots from the last N days within a radius.
  2. Night-light change: latest VIIRS month versus the 2019-2023 median (new lights in a dark area).

The verdict is deliberately conservative about what "nothing found" means: it is only reported as
"No anomaly detected" when the checks actually ran. If Earth Engine could not be reached the answer
is "Could not check", never a false all-clear. Satellites also miss small or cloud-covered fires,
so a clear result means "nothing detected", not "certainly safe".
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional


DEFAULT_RADIUS_KM = 10.0
DEFAULT_DAYS = 10
LIGHT_DELTA_FLAG = 5.0          # nW/cm²/sr rise that counts as a new light source
MAX_HOTSPOTS = 500

FIRE = "fire"
LIGHT = "light"
CLEAR = "clear"
UNKNOWN = "unknown"


def hotspot_distance_km(lat: float, lon: float, h: Dict) -> float:
    p1, p2 = math.radians(lat), math.radians(h["lat"])
    dp, dl = p2 - p1, math.radians(h["lon"] - lon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def decide(hotspots: Optional[List[Dict]], light_delta: Optional[float], errors: List[str]) -> Dict:
    """
    Verdict from the evidence. `hotspots` None means the fire check did not run; likewise `light_delta`.
    Returns {status, headline, detail}.
    """
    if hotspots is None and light_delta is None:
        return {"status": UNKNOWN, "headline": "Could not check this place",
                "detail": "Neither satellite check could run" + (f" ({errors[0]})." if errors else ".")
                          + " This is not an all-clear."}
    if hotspots:
        n = len(hotspots)
        hottest = max((h.get("temp_k") or 0) for h in hotspots)
        near = min(h["distance_km"] for h in hotspots)
        return {"status": FIRE, "headline": f"Fire detected: {n} thermal hotspot{'s' if n != 1 else ''}",
                "detail": f"Nearest is {near:.1f} km away" + (f", hottest pixel about {hottest:.0f} K." if hottest else ".")}
    if light_delta is not None and light_delta >= LIGHT_DELTA_FLAG:
        return {"status": LIGHT, "headline": "Night-light anomaly",
                "detail": f"No fire detected, but night lights here rose by about {light_delta:.1f} nW/cm²/sr "
                          "compared with the 2019–2023 normal. Could be new buildings, a camp or industry."}
    partial = hotspots is None or light_delta is None
    detail = "No thermal hotspots and no unusual new night lights were found."
    if partial:
        detail = ("No anomaly was found by the check that ran, but the other check could not run"
                  + (f" ({errors[0]})." if errors else "."))
    return {"status": CLEAR, "headline": "No anomaly detected", "detail": detail}


def _fire_from_firms(lat: float, lon: float, radius_km: float, days: int) -> List[Dict]:
    """Real hotspots from the NASA FIRMS API (needs FIRMS_MAP_KEY), nearest first, within the radius."""
    import firms
    from weather import bounds_around

    raw = firms.fetch_hotspots(bounds_around(lat, lon, radius_km), days=days)
    hs = []
    for h in raw:
        d = hotspot_distance_km(lat, lon, h)
        if d <= radius_km:
            hs.append({"lat": h["lat"], "lon": h["lon"], "temp_k": h["temp_k"] or 0.0, "distance_km": d,
                       "frp": h.get("frp"), "when": f"{h['date']} {h['time'][:2]}:{h['time'][2:]} UTC",
                       "satellite": h.get("satellite", ""), "confidence": h.get("confidence", "")})
    return sorted(hs, key=lambda h: h["distance_km"])


def _fire_from_ee(lat: float, lon: float, radius_km: float, days: int, today: date) -> List[Dict]:
    """Fallback: the FIRMS collection inside Earth Engine (daily mosaic, coarser than the API)."""
    import ee

    from gee_gateway import connect_gee
    from reliability import call_with_retry

    connect_gee()
    region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
    fires = (ee.ImageCollection("FIRMS")
             .filterDate((today - timedelta(days=days)).isoformat(), (today + timedelta(days=1)).isoformat())
             .select("T21").max())
    img = ee.Image.pixelLonLat().addBands(fires.rename("t")).updateMask(fires.gt(0))
    res = call_with_retry(lambda: img.reduceRegion(
        reducer=ee.Reducer.toList(), geometry=region, scale=1000, maxPixels=1_000_000, bestEffort=True).getInfo())
    hs = []
    for lo, la, tk in list(zip(res.get("longitude") or [], res.get("latitude") or [], res.get("t") or []))[:MAX_HOTSPOTS]:
        h = {"lat": float(la), "lon": float(lo), "temp_k": float(tk)}
        h["distance_km"] = hotspot_distance_km(lat, lon, h)
        if h["distance_km"] <= radius_km:
            hs.append(h)
    return sorted(hs, key=lambda h: h["distance_km"])


def _light_from_ee(lat: float, lon: float, radius_km: float) -> Optional[float]:
    import ee

    from gee_gateway import connect_gee
    from pipeline import BASELINE_RANGE, VIIRS_COLLECTION, latest_viirs_date, rolling_window
    from reliability import call_with_retry

    connect_gee()
    point = ee.Geometry.Point([lon, lat])
    coll = ee.ImageCollection(VIIRS_COLLECTION).select("avg_rad")
    start, end, _ = rolling_window(latest_viirs_date(ee, (lon - 1, lat - 1, lon + 1, lat + 1)))
    base = coll.filterDate(*BASELINE_RANGE).median()
    cur = coll.filterDate(start.isoformat(), end.isoformat()).mean()
    buf = point.buffer(min(radius_km, 3) * 1000)
    res = call_with_retry(lambda: cur.subtract(base).reduceRegion(
        reducer=ee.Reducer.max(), geometry=buf, scale=500, maxPixels=1_000_000, bestEffort=True).getInfo())
    v = next(iter(res.values()), None)
    return None if v is None else float(v)


def _default_fetch(lat: float, lon: float, radius_km: float, days: int, today: date) -> Dict:
    """
    Real data from whatever is configured: fire from NASA FIRMS when FIRMS_MAP_KEY is set (else from
    Earth Engine), night-light change from Earth Engine. A source that fails is reported, not hidden.
    """
    import apis

    out: Dict = {"hotspots": None, "light_delta": None, "errors": [], "fire_source": None}
    if apis.firms_key():
        try:
            out["hotspots"], out["fire_source"] = _fire_from_firms(lat, lon, radius_km, days), "NASA FIRMS API"
        except Exception as e:
            out["errors"].append(apis.redact(f"FIRMS fire check: {type(e).__name__}: {str(e)[:80]}"))
    if out["hotspots"] is None:
        try:
            out["hotspots"], out["fire_source"] = _fire_from_ee(lat, lon, radius_km, days, today), "Earth Engine FIRMS"
        except Exception as e:
            out["errors"].append(apis.redact(f"fire check: {type(e).__name__}"))
    try:
        out["light_delta"] = _light_from_ee(lat, lon, radius_km)
    except Exception as e:
        out["errors"].append(apis.redact(f"night-light check: {type(e).__name__}"))
    return out


def check_place(lat: float, lon: float, radius_km: float = DEFAULT_RADIUS_KM, days: int = DEFAULT_DAYS,
                fetcher: Optional[Callable[..., Dict]] = None, today: Optional[date] = None) -> Dict:
    """Run the checks and return {status, headline, detail, hotspots, light_delta, radius_km, days, errors}."""
    fetch = fetcher or _default_fetch
    try:
        data = fetch(lat, lon, radius_km, days, today or date.today())
    except Exception as e:
        data = {"hotspots": None, "light_delta": None, "errors": [f"{type(e).__name__}: {str(e)[:100]}"]}
    verdict = decide(data.get("hotspots"), data.get("light_delta"), data.get("errors", []))
    return {**verdict, "hotspots": data.get("hotspots") or [], "light_delta": data.get("light_delta"),
            "fire_source": data.get("fire_source"),
            "radius_km": radius_km, "days": days, "errors": data.get("errors", [])}
