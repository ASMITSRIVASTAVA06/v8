"""
TerraVision AI — weather hazard engine.

Turns the hourly forecast from Open-Meteo (free, no key) into risk levels for any point or grid of
points: thunderstorms, heavy rain, damaging wind and heat. Everything except `fetch_forecast`
is plain Python so it can be tested without a network.

Levels:  0 Low · 1 Watch · 2 Warning · 3 Severe

Thresholds follow common practice (India Meteorological Department rainfall classes for 24 h:
heavy 64.5 mm, very heavy 115.6 mm, extremely heavy 204.5 mm; hurricane-force gusts from 118 km/h),
but this is an indication from model forecasts, NOT an official warning. Always follow IMD, NHC,
JTWC or your national weather service for decisions that affect safety.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import apis
from reliability import call_with_retry, get_logger

HOURLY = ["temperature_2m", "precipitation", "wind_gusts_10m", "cape", "weather_code"]
BATCH = 20                      # locations per request
CACHE_TTL_S = 30 * 60
MAX_GRID_POINTS = 100

LEVELS = ["Low", "Watch", "Warning", "Severe"]
LEVEL_COLORS = {0: "#34d399", 1: "#facc15", 2: "#fb923c", 3: "#ef4444"}
HAZARD_LABELS = {"storm": "Thunderstorm", "rain": "Heavy rain", "wind": "Damaging wind", "heat": "Extreme heat",
                 "cyclone": "Cyclone"}

THUNDER_CODES = {95, 96, 99}
HAIL_CODES = {96, 99}


class WeatherError(RuntimeError):
    """The forecast service could not be reached or returned something unusable."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


# ── scoring ──────────────────────────────────────────────────────────────────

def _level(value: float, cuts: Sequence[float]) -> int:
    return sum(1 for c in cuts if value >= c)


def rolling_max_sum(values: Sequence[float], window: int = 24) -> Tuple[float, int]:
    """(largest sum over any `window` consecutive values, index where that window ends)."""
    vals = [float(v or 0) for v in values]
    if not vals:
        return 0.0, 0
    best, best_i, run = 0.0, 0, 0.0
    for i, v in enumerate(vals):
        run += v
        if i >= window:
            run -= vals[i - window]
        if run > best:
            best, best_i = run, i
    return round(best, 1), best_i


def _argmax(values: Sequence[float]) -> int:
    vals = [float(v or 0) for v in values]
    return max(range(len(vals)), key=vals.__getitem__) if vals else 0


def _clean(seq, n: int) -> List[float]:
    out = []
    for v in (seq or [])[:n]:
        try:
            f = float(v)
            out.append(0.0 if math.isnan(f) else f)
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def assess_point(hourly: Dict, hours: int = 72) -> Dict:
    """
    Hazard levels from one location's hourly forecast (Open-Meteo `hourly` object).
    Returns {level, hazards: {name: {level, value, unit, at}}, headline}.
    """
    times = (hourly.get("time") or [])[:hours]
    n = len(times) or hours
    temp = _clean(hourly.get("temperature_2m"), n)
    rain = _clean(hourly.get("precipitation"), n)
    gust = _clean(hourly.get("wind_gusts_10m"), n)
    cape = _clean(hourly.get("cape"), n)
    codes = [int(c) for c in _clean(hourly.get("weather_code"), n)]

    def at(i: int) -> str:
        return times[i] if 0 <= i < len(times) else ""

    hz: Dict[str, Dict] = {}

    # thunderstorm: stored energy (CAPE) plus an actual thunderstorm in the model's weather codes
    peak_cape = max(cape) if cape else 0.0
    thunder_idx = [i for i, c in enumerate(codes) if c in THUNDER_CODES]
    storm = _level(peak_cape, (1000, 2000, 3500))
    storm = min(storm, 1)                      # energy alone is only a Watch
    if thunder_idx:
        storm = max(storm, 1)
        if peak_cape >= 1000:
            storm = max(storm, 2)
        if any(codes[i] in HAIL_CODES for i in thunder_idx):
            storm = min(3, storm + 1)
        if peak_cape >= 3500:
            storm = min(3, max(storm, 3))
    hz["storm"] = {"level": storm, "value": round(peak_cape), "unit": "J/kg CAPE",
                   "at": at(thunder_idx[0]) if thunder_idx else at(_argmax(cape)),
                   "thunder_hours": len(thunder_idx)}

    total24, end_i = rolling_max_sum(rain, 24)
    peak_hour = max(rain) if rain else 0.0
    rain_level = max(_level(total24, (64.5, 115.6, 204.5)), 2 if peak_hour >= 30 else 0)
    hz["rain"] = {"level": rain_level, "value": total24, "unit": "mm in 24 h", "at": at(end_i),
                  "peak_hour_mm": round(peak_hour, 1)}

    peak_gust = max(gust) if gust else 0.0
    hz["wind"] = {"level": _level(peak_gust, (60, 90, 118)), "value": round(peak_gust), "unit": "km/h gusts",
                  "at": at(_argmax(gust))}

    peak_t = max(temp) if temp else 0.0
    hz["heat"] = {"level": _level(peak_t, (40, 43, 46)), "value": round(peak_t, 1), "unit": "°C", "at": at(_argmax(temp))}

    level = max(h["level"] for h in hz.values())
    return {"level": level, "hazards": hz, "headline": headline(hz)}


def headline(hazards: Dict[str, Dict]) -> str:
    """Short text naming the hazards at the top level, e.g. 'Warning: heavy rain, thunderstorm'."""
    top = max((h["level"] for h in hazards.values()), default=0)
    if top == 0:
        return "No significant weather hazard forecast"
    names = [HAZARD_LABELS.get(k, k).lower() for k, h in hazards.items() if h["level"] == top]
    return f"{LEVELS[top]}: " + ", ".join(names)


# ── grid ─────────────────────────────────────────────────────────────────────

def make_grid(bounds: Tuple[float, float, float, float], step_deg: float = 1.0,
              max_points: int = MAX_GRID_POINTS) -> List[Dict]:
    """
    Cell centres covering (min_lon, min_lat, max_lon, max_lat). The step grows if needed so the
    number of cells never exceeds `max_points` (keeps forecast requests small and polite).
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    step = max(0.1, float(step_deg))
    while True:
        cols = max(1, math.ceil((max_lon - min_lon) / step))
        rows = max(1, math.ceil((max_lat - min_lat) / step))
        if cols * rows <= max_points:
            break
        step *= 1.25
    sx, sy = (max_lon - min_lon) / cols, (max_lat - min_lat) / rows
    return [{"lat": round(min_lat + (r + .5) * sy, 4), "lon": round(min_lon + (c + .5) * sx, 4),
             "half_lat": sy / 2, "half_lon": sx / 2}
            for r in range(rows) for c in range(cols)]


def bounds_around(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


# ── fetching ─────────────────────────────────────────────────────────────────

_CACHE: Dict[Tuple, Tuple[float, Dict]] = {}


def _http_json(url: str, timeout: int = 25) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "TerraVisionAI/1.0 (forest monitoring research tool)",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _request(points: List[Dict], hours: int, http: Callable[[str], object]) -> List[Dict]:
    days = max(1, min(7, math.ceil(hours / 24)))
    q = {"latitude": ",".join(f"{p['lat']:.4f}" for p in points),
         "longitude": ",".join(f"{p['lon']:.4f}" for p in points),
         "hourly": ",".join(HOURLY), "forecast_days": days, "timezone": "UTC", "wind_speed_unit": "kmh"}
    ep = apis.weather_endpoint()
    if ep["key"]:
        q["apikey"] = ep["key"]
    data = call_with_retry(http, ep["url"] + "?" + urllib.parse.urlencode(q, safe=","))
    items = data if isinstance(data, list) else [data]
    if len(items) != len(points) or any(not isinstance(i, dict) or "hourly" not in i for i in items):
        reason = data.get("reason") if isinstance(data, dict) else None
        raise WeatherError(f"unexpected forecast response ({reason or 'shape mismatch'})")
    return [i["hourly"] for i in items]


def fetch_forecast(points: List[Dict], hours: int = 72,
                   http: Optional[Callable[[str], object]] = None, now: Optional[float] = None
                   ) -> Tuple[List[Optional[Dict]], List[str]]:
    """
    Hourly forecast for each point: (list aligned with `points` — None where a request failed, errors).
    Results are cached for 30 minutes per point so reloading the page does not re-query the service.
    """
    http = http or _http_json
    now = time.time() if now is None else now
    out: List[Optional[Dict]] = [None] * len(points)
    errors: List[str] = []
    todo: List[int] = []
    for i, p in enumerate(points):
        hit = _CACHE.get((round(p["lat"], 3), round(p["lon"], 3), hours))
        if hit and now - hit[0] < CACHE_TTL_S:
            out[i] = hit[1]
        else:
            todo.append(i)
    for s in range(0, len(todo), BATCH):
        idx = todo[s:s + BATCH]
        batch = [points[i] for i in idx]
        try:
            results = _request(batch, hours, http)
        except Exception as e:
            get_logger().warning("Forecast batch failed (%s); retrying points one by one", apis.redact(e))
            results = []
            for p in batch:                      # fall back: one point per request
                try:
                    results.append(_request([p], hours, http)[0])
                except Exception as e2:
                    results.append(None)
                    if len(errors) < 3:
                        errors.append(apis.redact(f"{type(e2).__name__}: {str(e2)[:100]}"))
        for i, r in zip(idx, results):
            out[i] = r
            if r is not None:
                _CACHE[(round(points[i]["lat"], 3), round(points[i]["lon"], 3), hours)] = (now, r)
    return out, errors


# ── area assessment ──────────────────────────────────────────────────────────

def assess_area(bounds: Tuple[float, float, float, float], step_deg: float = 1.0, hours: int = 72,
                http: Optional[Callable[[str], object]] = None, now: Optional[float] = None) -> Dict:
    """
    Forecast hazards over a rectangle: {cells: [{lat, lon, half_lat, half_lon, level, hazards, headline}],
    errors, hours, generated_at, worst}. Cells whose forecast failed are left out and counted in `missing`.
    """
    grid = make_grid(bounds, step_deg)
    forecasts, errors = fetch_forecast(grid, hours, http, now)
    cells = []
    for g, f in zip(grid, forecasts):
        if f is None:
            continue
        cells.append({**g, **assess_point(f, hours)})
    return {"cells": cells, "errors": errors, "hours": hours, "missing": len(grid) - len(cells),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "worst": max((c["level"] for c in cells), default=0)}


def flagged(cells: List[Dict], min_level: int = 1) -> List[Dict]:
    """Cells at or above `min_level`, most severe first."""
    return sorted((c for c in cells if c["level"] >= min_level), key=lambda c: (-c["level"], c["lat"], c["lon"]))
