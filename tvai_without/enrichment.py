"""
TerraVision AI — satellite cross-checks for each alert (Google Earth Engine).

For every alert point this adds independent evidence from other datasets:

  protected area   WCMC/WDPA  (World Database on Protected Areas polygons)
  built-up land    ESA WorldCover 10 m  (urban lights vs. forest)
  vegetation       Sentinel-2 NDVI, last 90 days vs. same season a year earlier
  forest loss      Hansen Global Forest Change (loss in the last two years)
  active fire      NASA FIRMS thermal anomalies (last 60 days)

Each layer is fetched independently: if one dataset is unavailable or errors
out, the others still apply and the alert simply lacks that field (None).
Which layers worked is returned so the UI can say so honestly.

Requires an initialised Earth Engine session (gee_gateway.connect_gee()).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Dict, List, Tuple

from reliability import call_with_retry, get_logger

CHUNK = 400            # points per Earth Engine request (keeps replies small)
PIXEL_BUFFER_M = 250   # half of a 500 m VIIRS pixel

HANSEN_IDS = [         # newest first; the first one that exists is used
    "UMD/hansen/global_forest_change_2025_v1_13",
    "UMD/hansen/global_forest_change_2024_v1_12",
    "UMD/hansen/global_forest_change_2023_v1_11",
]

LAYERS = ["protected_area", "land_cover", "vegetation", "forest_loss", "active_fire"]


def _points(ee, alerts: List[Dict], start: int, end: int):
    feats = [
        ee.Feature(ee.Geometry.Point([a["lon"], a["lat"]]), {"i": start + k})
        for k, a in enumerate(alerts[start:end])
    ]
    return ee.FeatureCollection(feats)


def _buffered(ee, fc):
    return fc.map(lambda f: f.setGeometry(f.geometry().buffer(PIXEL_BUFFER_M)))


def _by_index(info: Dict) -> Dict[int, Dict]:
    return {f["properties"]["i"]: f["properties"] for f in info["features"]}


def _num(v):
    return None if v is None else float(v)


# ── individual layers: each returns {alert_index: {field: value}} ───────────

def _protected_area(ee, pts) -> Dict[int, Dict]:
    # only protection that is legally in force, and not purely marine areas
    wdpa = (ee.FeatureCollection("WCMC/WDPA/current/polygons")
            .filter(ee.Filter.inList("STATUS", ["Designated", "Inscribed", "Established"])))
    join = ee.Join.saveFirst("pa")
    joined = join.apply(pts, wdpa, ee.Filter.intersects(leftField=".geo", rightField=".geo", maxError=100))

    def tag(f):
        missing = ee.Algorithms.IsEqual(f.get("pa"), None)
        name = ee.Algorithms.If(missing, "", ee.Feature(f.get("pa")).get("NAME"))
        return ee.Feature(None, {"i": f.get("i"), "pa_name": name,
                                 "pa_found": ee.Algorithms.If(missing, 0, 1)})

    info = joined.map(tag).getInfo()
    out = {}
    for i, p in _by_index(info).items():
        found = bool(p.get("pa_found"))
        out[i] = {"in_protected_area": found, "protected_area": p.get("pa_name") or None}
    return out


def _land_cover(ee, pts) -> Dict[int, Dict]:
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    img = ee.Image.cat([wc.eq(50).rename("builtup"), wc.eq(10).rename("tree")])
    info = img.reduceRegions(collection=_buffered(ee, pts), reducer=ee.Reducer.mean(), scale=100).getInfo()
    out = {i: {"builtup_fraction": _num(p.get("builtup")), "tree_fraction": _num(p.get("tree"))}
           for i, p in _by_index(info).items()}
    # wider 1 km view: is this pixel inside a forest or in farmland / town?
    ctx = ee.Image.cat([wc.eq(10).rename("tree_ctx"), wc.eq(40).rename("crop_ctx")])
    wide = pts.map(lambda f: f.setGeometry(f.geometry().buffer(1000)))
    info2 = ctx.reduceRegions(collection=wide, reducer=ee.Reducer.mean(), scale=100).getInfo()
    for i, p in _by_index(info2).items():
        out.setdefault(i, {}).update(tree_context=_num(p.get("tree_ctx")), crop_context=_num(p.get("crop_ctx")))
    return out


def _ndvi_composite(ee, start: date, end: date):
    def prep(img):
        scl = img.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(1))
        return img.updateMask(clear).normalizedDifference(["B8", "B4"]).rename("ndvi")
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start.isoformat(), end.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .map(prep).median())


def _vegetation(ee, pts, today: date) -> Dict[int, Dict]:
    recent = _ndvi_composite(ee, today - timedelta(days=90), today)
    prior = _ndvi_composite(ee, today - timedelta(days=90 + 365), today - timedelta(days=365))
    change = recent.subtract(prior).rename("ndvi_change")
    info = change.reduceRegions(collection=_buffered(ee, pts), reducer=ee.Reducer.mean(), scale=30).getInfo()
    return {i: {"ndvi_change": _num(p.get("ndvi_change", p.get("mean")))}
            for i, p in _by_index(info).items()}


def _hansen_image(ee):
    """Newest Hansen asset that exists, with the last year it covers (2000 + N)."""
    for hid in HANSEN_IDS:
        try:
            img = ee.Image(hid)
            img.bandNames().getInfo()   # raises if the asset does not exist
            year = int(hid.split("global_forest_change_")[1][:4])
            return img, year
        except Exception:
            continue
    raise RuntimeError("No Hansen Global Forest Change asset available")


def _forest_loss(ee, pts) -> Dict[int, Dict]:
    h, last_year = _hansen_image(ee)
    loss_year = h.select("lossyear")                       # 1..N means year 2000+N
    recent = loss_year.gte(last_year - 2000 - 1).And(loss_year.gt(0))   # last two years of data
    forested = h.select("treecover2000").gte(30)           # only land that was forest in 2000
    img = recent.And(forested).rename("loss")
    info = img.reduceRegions(collection=_buffered(ee, pts), reducer=ee.Reducer.mean(), scale=30).getInfo()
    return {i: {"tree_loss_fraction": _num(p.get("loss", p.get("mean")))}
            for i, p in _by_index(info).items()}


def _active_fire(ee, pts, today: date) -> Dict[int, Dict]:
    fires = (ee.ImageCollection("FIRMS")
             .filterDate((today - timedelta(days=60)).isoformat(), today.isoformat())
             .select("T21").max().gt(0).unmask(0).rename("fire"))
    buffered = pts.map(lambda f: f.setGeometry(f.geometry().buffer(1500)))
    info = fires.reduceRegions(collection=buffered, reducer=ee.Reducer.max(), scale=1000).getInfo()
    return {i: {"fire_detected": bool(p.get("fire", p.get("max")) or 0)} for i, p in _by_index(info).items()}


LAYER_FUNCS: List[Tuple[str, Callable]] = [
    ("protected_area", lambda ee, pts, today: _protected_area(ee, pts)),
    ("land_cover",     lambda ee, pts, today: _land_cover(ee, pts)),
    ("vegetation",     _vegetation),
    ("forest_loss",    lambda ee, pts, today: _forest_loss(ee, pts)),
    ("active_fire",    _active_fire),
]


def add_context(
    alerts: List[Dict],
    progress: Callable[[str], None] = lambda msg: None,
    today: date | None = None,
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Add cross-check fields to every alert in place.
    Returns (alerts, layers_ok, notes). A layer counts as "ok" only if it worked
    for every chunk; failures are described in `notes`, never raised.
    """
    import ee  # imported lazily so the rest of the app works without Earth Engine

    today = today or date.today()
    if not alerts:
        return alerts, list(LAYERS), []

    failed: Dict[str, str] = {}
    n = len(alerts)
    for name, fn in LAYER_FUNCS:
        progress(f"Cross-checking alerts: {name.replace('_', ' ')}…")
        for start in range(0, n, CHUNK):
            if name in failed:
                break
            try:
                pts = _points(ee, alerts, start, min(start + CHUNK, n))
                result = call_with_retry(fn, ee, pts, today)
                for i, fields in result.items():
                    alerts[i].update(fields)
            except Exception as e:  # keep scanning with the remaining layers
                failed[name] = f"{type(e).__name__}: {str(e)[:160]}"
                get_logger().warning("Cross-check layer '%s' failed: %s", name, failed[name])

    ok = [l for l in LAYERS if l not in failed]
    notes = [f"Cross-check '{l}' unavailable ({msg})" for l, msg in failed.items()]
    return alerts, ok, notes
