"""
TerraVision AI — find a place.

Turns what someone types into coordinates:
  * "27.2046, 77.4977", "27.2046 N 77.4977 E", "27°12'16\"N 77°29'51\"E"  -> parsed locally, no network
  * "Sariska Tiger Reserve", "Daintree, Queensland"                     -> OpenStreetMap Nominatim

Nominatim is a free public service with a usage policy (one request per second, an identifying
User-Agent, no bulk use), so results are meant to be cached by the caller and this module sends
one request per search. Set TERRAVISION_CONTACT to your e-mail or site so the operators can reach you.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

import apis

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
TIMEOUT_S = 12

_NUM = r"[-+]?\d+(?:\.\d+)?"
_DMS = re.compile(
    rf"(?P<d>\d+(?:\.\d+)?)\s*[°d]\s*(?:(?P<m>\d+(?:\.\d+)?)\s*['′m]\s*)?(?:(?P<s>\d+(?:\.\d+)?)\s*[\"″s]\s*)?(?P<h>[NSEWnsew])")


class GeocodeError(RuntimeError):
    """The place could not be looked up (network problem or service refused)."""


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _dms_value(m: "re.Match") -> float:
    v = float(m["d"]) + float(m["m"] or 0) / 60 + float(m["s"] or 0) / 3600
    return -v if m["h"].upper() in "SW" else v


def parse_coordinates(text: str) -> Optional[Dict]:
    """
    {'lat', 'lon', 'name'} if `text` is a coordinate pair, else None.
    Accepts decimal degrees (lat first), hemisphere letters, and degrees/minutes/seconds.
    """
    t = (text or "").strip()
    if not t:
        return None

    dms = list(_DMS.finditer(t))
    if len(dms) == 2:
        a, b = dms
        if a["h"].upper() in "NS" and b["h"].upper() in "EW":
            lat, lon = _dms_value(a), _dms_value(b)
        elif a["h"].upper() in "EW" and b["h"].upper() in "NS":
            lat, lon = _dms_value(b), _dms_value(a)
        else:
            return None
        return {"lat": lat, "lon": lon, "name": t} if _valid(lat, lon) else None

    m = re.fullmatch(rf"\s*({_NUM})\s*°?\s*([NSns])?\s*[,;\s]\s*({_NUM})\s*°?\s*([EWew])?\s*", t)
    if not m:
        return None
    lat, lon = float(m[1]), float(m[3])
    if m[2] and m[2].upper() == "S":
        lat = -abs(lat)
    if m[4] and m[4].upper() == "W":
        lon = -abs(lon)
    return {"lat": lat, "lon": lon, "name": t} if _valid(lat, lon) else None


def _user_agent() -> str:
    contact = os.environ.get("TERRAVISION_CONTACT", "").strip()
    return "TerraVisionAI/1.0 (forest monitoring research tool" + (f"; {contact})" if contact else ")")


def search_place(query: str, limit: int = 5, timeout: int = TIMEOUT_S) -> List[Dict]:
    """
    Candidate places for `query`, best first: [{'name', 'lat', 'lon', 'kind', 'bbox'}].
    Coordinates typed directly are returned without any network call.
    Raises GeocodeError if the service cannot be reached; returns [] if nothing matches.
    """
    q = (query or "").strip()
    if not q:
        return []
    direct = parse_coordinates(q)
    if direct:
        return [{**direct, "kind": "coordinates", "bbox": None}]

    ep = apis.geocoder_endpoint()
    params = {"q": q, "format": "json", "limit": max(1, min(limit, 10)), "addressdetails": 0}
    if ep["key"]:
        params["key"] = ep["key"]
    url = ep["url"] + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent(), "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise GeocodeError(apis.redact(f"{type(e).__name__}: {str(e)[:120]}")) from None

    out = []
    for r in data if isinstance(data, list) else []:
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not _valid(lat, lon):
            continue
        bb = r.get("boundingbox")
        bbox = None
        if bb and len(bb) == 4:
            try:
                s, n, w, e = (float(x) for x in bb)        # Nominatim order: south, north, west, east
                bbox = (w, s, e, n)
            except (TypeError, ValueError):
                bbox = None
        out.append({"name": r.get("display_name") or q, "lat": lat, "lon": lon,
                    "kind": r.get("type") or r.get("category") or r.get("class") or "place", "bbox": bbox})
    return out
