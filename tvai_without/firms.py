"""
TerraVision AI — real fire hotspots from NASA FIRMS.

Uses the FIRMS "area" API (free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/):
VIIRS and MODIS thermal detections, usually available within a few hours of the satellite pass.

  GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/<SOURCE>/<west,south,east,north>/<days>

The service answers with CSV on success and with a short plain-text message (for example an
invalid key or an exceeded quota) on failure; both cases are handled and a key is never included
in an error message. Days are limited to 1-5 per request.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

import apis
from reliability import call_with_retry

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
MAX_DAYS = 5
CACHE_TTL_S = 10 * 60
_CACHE: Dict[Tuple, Tuple[float, List[Dict]]] = {}


class FirmsError(RuntimeError):
    """FIRMS refused or could not answer the request."""


def _http_text(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "TerraVisionAI/1.0 (forest monitoring research tool)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_csv(text: str, source: str = "") -> List[Dict]:
    """Hotspots from a FIRMS CSV; raises FirmsError if `text` is a service message rather than data."""
    body = (text or "").strip()
    if not body:
        return []
    first = body.splitlines()[0].lower()
    if "latitude" not in first:
        raise FirmsError(apis.redact(body[:120]))
    out = []
    for r in csv.DictReader(io.StringIO(body)):
        lat, lon = _num(r.get("latitude")), _num(r.get("longitude"))
        if lat is None or lon is None:
            continue
        temp = _num(r.get("bright_ti4")) or _num(r.get("brightness"))     # VIIRS I-4 band, or MODIS band 21/22
        out.append({"lat": lat, "lon": lon, "temp_k": temp, "frp": _num(r.get("frp")),
                    "confidence": str(r.get("confidence", "")).strip().lower(),
                    "date": r.get("acq_date", ""), "time": str(r.get("acq_time", "")).zfill(4),
                    "satellite": r.get("satellite", ""), "daynight": r.get("daynight", ""), "source": source})
    return out


def is_low_confidence(h: Dict) -> bool:
    """VIIRS marks low confidence with 'l'; MODIS uses a 0-100 number (below 30 is low)."""
    c = h.get("confidence", "")
    if c in ("l", "low"):
        return True
    n = _num(c)
    return n is not None and n < 30


def fetch_hotspots(bounds: Tuple[float, float, float, float], days: int = 2,
                   sources: Optional[List[str]] = None, key: Optional[str] = None,
                   http: Optional[Callable[[str], str]] = None, drop_low_confidence: bool = True,
                   now: Optional[float] = None) -> List[Dict]:
    """
    All hotspots in the (west, south, east, north) box over the last `days` days (1-5), from every source,
    de-duplicated. Raises FirmsError when no MAP_KEY is configured or every source fails.
    """
    key = key or apis.firms_key()
    if not key:
        raise FirmsError("FIRMS_MAP_KEY is not set")
    http = http or _http_text
    now = time.time() if now is None else now
    days = max(1, min(MAX_DAYS, int(days)))
    sources = sources or [s.strip() for s in apis.get_setting("FIRMS_SOURCES").split(",") if s.strip()] or DEFAULT_SOURCES
    area = ",".join(f"{v:.4f}" for v in bounds)

    ck = (area, days, tuple(sources), drop_low_confidence)
    hit = _CACHE.get(ck)
    if hit and now - hit[0] < CACHE_TTL_S:
        return hit[1]

    seen, out, errors, ok = set(), [], [], 0
    for src in sources:
        try:
            text = call_with_retry(http, f"{BASE}/{key}/{src}/{area}/{days}")
            rows = parse_csv(text, src)
        except Exception as e:
            errors.append(f"{src}: {apis.redact(f'{type(e).__name__}: {str(e)[:80]}', [key])}")
            continue
        ok += 1
        for h in rows:
            ident = (round(h["lat"], 3), round(h["lon"], 3), h["date"], h["time"][:3])
            if ident in seen or (drop_low_confidence and is_low_confidence(h)):
                continue
            seen.add(ident)
            out.append(h)
    if ok == 0:
        raise FirmsError("; ".join(errors) or "no source answered")
    _CACHE[ck] = (now, out)
    return out
