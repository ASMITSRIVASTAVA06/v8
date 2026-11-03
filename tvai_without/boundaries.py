"""
TerraVision AI — real region boundaries.

A rectangle is not a forest. This module gives every region its actual footprint:

  1. OFFICIAL boundary (best): drop a GeoJSON file named after the region into ./boundaries/,
     e.g. boundaries/aravalli_hills_india.geojson — reserved / protected forest limits from
     your forest department, Survey of India, FSI, Protected Planet or OpenStreetMap. When
     present it is authoritative: only land inside it counts as the region.
  2. WDPA (automatic): designated, terrestrial protected-area polygons from the World Database
     on Protected Areas, restricted to the region's search box AND its countries, fetched from
     Earth Engine and cached in ./data_cache/boundaries/.

Alerts are then tested against these polygons locally (point-in-polygon, holes respected), so
the check works offline and the same polygons can be drawn on the map.

Needs only NumPy; Earth Engine is used only to fetch the WDPA polygons.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from regions import REGION_SPECS, region_slug

PROJECT_DIR = Path(__file__).parent
OFFICIAL_DIR = PROJECT_DIR / "boundaries"
CACHE_DIR = PROJECT_DIR / "data_cache" / "boundaries"

WDPA_ASSET = "WCMC/WDPA/current/polygons"
# WDPA STATUS values meaning the protection is legally in force (Proposed / Not Reported are excluded)
DESIGNATED_STATUS = ["Designated", "Inscribed", "Established"]
PROPERTIES = ["NAME", "DESIG_ENG", "IUCN_CAT", "MARINE", "GIS_AREA"]
MAX_FEATURES = 800            # largest N protected areas per region (keeps the map and checks fast)
FETCH_CHUNK = 150
CACHE_MAX_AGE_DAYS = 90

Progress = Callable[[str], None]


# ── paths and loading ────────────────────────────────────────────────────────

def official_path(region: str) -> Path:
    return OFFICIAL_DIR / f"{region_slug(region)}.geojson"


def cache_path(region: str) -> Path:
    return CACHE_DIR / f"{region_slug(region)}.json"


def _features_from_geojson(gj: Dict) -> List[Dict]:
    """Accept FeatureCollection, Feature or bare geometry; keep only Polygon / MultiPolygon."""
    kind = gj.get("type")
    if kind == "FeatureCollection":
        feats = gj.get("features", [])
    elif kind == "Feature":
        feats = [gj]
    elif kind in ("Polygon", "MultiPolygon"):
        feats = [{"type": "Feature", "properties": {}, "geometry": gj}]
    else:
        feats = []
    out = []
    for f in feats:
        g = (f or {}).get("geometry") or {}
        if g.get("type") in ("Polygon", "MultiPolygon") and g.get("coordinates"):
            out.append({"type": "Feature", "properties": f.get("properties") or {}, "geometry": g})
    return out


def load_boundary(region: str) -> Optional[Dict]:
    """
    The best boundary available on disk, or None.
    Returns {"source": "official" | "wdpa", "features": [...], "fetched_at": str, "truncated": bool, "total": int}.
    """
    op = official_path(region)
    if op.exists():
        try:
            feats = _features_from_geojson(json.loads(op.read_text(encoding="utf-8")))
            if feats:
                return {"source": "official", "features": feats, "fetched_at": "", "truncated": False,
                        "total": len(feats)}
        except (OSError, ValueError):
            pass   # fall through to the automatic boundary rather than failing the scan
    cp = cache_path(region)
    if cp.exists():
        try:
            d = json.loads(cp.read_text(encoding="utf-8"))
            feats = _features_from_geojson({"type": "FeatureCollection", "features": d.get("features", [])})
            if feats:
                return {"source": "wdpa", "features": feats, "fetched_at": d.get("fetched_at", ""),
                        "truncated": bool(d.get("truncated")), "total": int(d.get("total", len(feats)))}
        except (OSError, ValueError):
            pass
    return None


def _is_stale(boundary: Dict, max_age_days: int = CACHE_MAX_AGE_DAYS) -> bool:
    if boundary["source"] != "wdpa":
        return False
    try:
        age = datetime.now() - datetime.strptime(boundary["fetched_at"], "%Y-%m-%d %H:%M")
    except ValueError:
        return True
    return age > timedelta(days=max_age_days)


# ── geometry ─────────────────────────────────────────────────────────────────

def _rings(geometry: Dict) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """[(outer_ring (N,2) lon/lat, [hole rings])] for a Polygon or MultiPolygon."""
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    out = []
    for poly in polys:
        if not poly:
            continue
        outer = np.asarray(poly[0], dtype=float)[:, :2]
        holes = [np.asarray(h, dtype=float)[:, :2] for h in poly[1:] if len(h) >= 4]
        if len(outer) >= 4:
            out.append((outer, holes))
    return out


def _in_ring(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for many points against one ring."""
    inside = np.zeros(lon.shape, dtype=bool)
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        if ay == by:
            continue
        crosses = (ay > lat) != (by > lat)
        if not crosses.any():
            continue
        x_at = (bx - ax) * (lat - ay) / (by - ay) + ax
        inside ^= crosses & (lon < x_at)
    return inside


def points_inside(boundary: Dict, lats: Iterable[float], lons: Iterable[float]) -> Tuple[np.ndarray, List[str]]:
    """
    For each point: is it inside any polygon of the boundary (holes excluded), and the name of
    the first polygon that contains it ("" if none).
    """
    lat = np.asarray(list(lats), dtype=float)
    lon = np.asarray(list(lons), dtype=float)
    inside = np.zeros(lat.shape, dtype=bool)
    names = [""] * len(lat)
    for feat in boundary["features"]:
        name = str((feat.get("properties") or {}).get("NAME") or (feat.get("properties") or {}).get("name") or "")
        for outer, holes in _rings(feat["geometry"]):
            minx, miny = outer.min(axis=0)
            maxx, maxy = outer.max(axis=0)
            cand = np.where(~inside & (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy))[0]
            if cand.size == 0:
                continue
            hit = _in_ring(lon[cand], lat[cand], outer)
            for hole in holes:
                if hit.any():
                    hit &= ~_in_ring(lon[cand], lat[cand], hole)
            for k in cand[hit]:
                inside[k] = True
                names[k] = name
    return inside, names


def annotate_alerts(alerts: List[Dict], boundary: Optional[Dict], fill_protected_area: bool = False) -> int:
    """
    Add boundary membership to alerts in place. Returns how many alerts were inside.

    official boundary -> in_official_boundary (True/False) and boundary_name
    wdpa boundary     -> only if `fill_protected_area` (used when the Earth Engine check was unavailable):
                         fills in_protected_area / protected_area for alerts that lack them
    """
    if not boundary or not alerts:
        return 0
    inside, names = points_inside(boundary, [a["lat"] for a in alerts], [a["lon"] for a in alerts])
    for a, ok, name in zip(alerts, inside, names):
        if boundary["source"] == "official":
            a["in_official_boundary"] = bool(ok)
            a["boundary_name"] = name or None
        elif fill_protected_area and a.get("in_protected_area") is None:
            a["in_protected_area"] = bool(ok)
            a["protected_area"] = name or None
    return int(inside.sum())


def map_geojson(boundary: Optional[Dict], max_features: int = 400) -> Optional[Dict]:
    """A FeatureCollection ready for folium.GeoJson (biggest areas first, capped for speed)."""
    if not boundary:
        return None
    return {"type": "FeatureCollection", "features": boundary["features"][:max_features]}


def describe(boundary: Optional[Dict]) -> str:
    """One line for the UI: what defines this region right now."""
    if not boundary:
        return "Search box only (no boundary loaded yet — a scan will fetch protected-area polygons)."
    n = len(boundary["features"])
    if boundary["source"] == "official":
        return f"Official boundary file ({n} area{'s' if n != 1 else ''})."
    more = f" of {boundary['total']}" if boundary.get("truncated") else ""
    return (f"{n}{more} designated protected areas from WDPA"
            + (f", fetched {boundary['fetched_at']}" if boundary.get("fetched_at") else "") + ".")


# ── fetching WDPA polygons from Earth Engine ─────────────────────────────────

def fetch_wdpa_boundary(region: str, progress: Progress = lambda m: None) -> Dict:
    """
    Download the region's designated terrestrial protected areas, simplified, and cache them.
    Raises on failure; callers should use ensure_boundary() which handles that gracefully.
    """
    import ee

    from reliability import call_with_retry

    spec = REGION_SPECS[region]
    box = ee.Geometry.Rectangle(list(spec["bounds"]))
    fc = (ee.FeatureCollection(WDPA_ASSET)
          .filterBounds(box)
          .filter(ee.Filter.inList("ISO3", spec["iso3"]))
          .filter(ee.Filter.inList("STATUS", DESIGNATED_STATUS)))
    total = int(call_with_retry(fc.size().getInfo))
    progress(f"Fetching {min(total, MAX_FEATURES)} of {total} protected areas for the boundary…")

    fc = (fc.sort("GIS_AREA", False).limit(MAX_FEATURES)
            .map(lambda f: f.select(PROPERTIES).setGeometry(
                f.geometry().intersection(box, 1000).simplify(500))))
    feats: List[Dict] = []
    for offset in range(0, min(total, MAX_FEATURES), FETCH_CHUNK):
        chunk = ee.FeatureCollection(fc.toList(FETCH_CHUNK, offset))
        feats += call_with_retry(chunk.getInfo)["features"]

    kept = []
    for f in _features_from_geojson({"type": "FeatureCollection", "features": feats}):
        if str(f["properties"].get("MARINE", "0")).split(".")[0] != "2":      # drop fully marine areas
            kept.append(f)
    data = {
        "source": "wdpa", "region": region, "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": total, "truncated": total > MAX_FEATURES, "features": kept,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path(region).with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(cache_path(region))
    return load_boundary(region) or {**data}


def ensure_boundary(region: str, progress: Progress = lambda m: None,
                    refresh: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Return (boundary, note). Uses an official file if present, otherwise a fresh-enough cached
    WDPA boundary, otherwise tries to fetch one. A failed fetch is reported in `note`, never raised;
    an older cached boundary is still used in that case.
    """
    existing = load_boundary(region)
    if existing and existing["source"] == "official":
        return existing, None
    if existing and not refresh and not _is_stale(existing):
        return existing, None
    try:
        return fetch_wdpa_boundary(region, progress), None
    except Exception as e:
        note = f"Could not fetch the protected-area boundary ({type(e).__name__}: {str(e)[:120]})."
        if existing:
            return existing, note + " Using the previously saved boundary."
        return None, note + " The region is limited to its search box for this scan."
