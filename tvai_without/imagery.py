"""
TerraVision AI — before / after satellite imagery for one alert.

Night-light change says "something changed here"; a reviewer decides in seconds by
looking at the ground. For a chosen alert this builds two Sentinel-2 true-colour
thumbnails of the same area and the same season, one year apart, with the ~500 m
VIIRS pixel outlined. Fetched on demand (one alert at a time) so it never slows a scan.

Requires an initialised Earth Engine session (gee_gateway.connect_gee()).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Tuple

from alert_tools import pixel_footprint
from reliability import call_with_retry

WINDOW_DAYS = 120          # long enough to get cloud-free scenes in monsoon regions
HALF_WIDTH_M = 1200        # thumbnail covers about 2.4 km x 2.4 km around the alert
THUMB_PX = 512


def image_windows(today: date, window_days: int = WINDOW_DAYS) -> Dict[str, Tuple[date, date]]:
    """Same calendar window, now and one year earlier, so seasons match."""
    after_end = today
    after_start = today - timedelta(days=window_days)
    return {
        "before": (after_start - timedelta(days=365), after_end - timedelta(days=365)),
        "after": (after_start, after_end),
    }


def _composite(ee, start: date, end: date, region):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(start.isoformat(), end.isoformat())
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))

    def mask_clouds(img):
        scl = img.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(1))
        return img.updateMask(clear)

    return col.map(mask_clouds).median().select(["B4", "B3", "B2"]), col.size()


def before_after_urls(lat: float, lon: float, today: date | None = None) -> Dict:
    """
    Returns {"before": url, "after": url, "labels": {...}, "scenes": {...}} or raises.
    Each URL is a short-lived JPEG thumbnail served by Earth Engine.
    """
    import ee

    today = today or date.today()
    region = ee.Geometry.Point([lon, lat]).buffer(HALF_WIDTH_M).bounds()
    (s, w), (n, e) = pixel_footprint(lat, lon)
    outline = (ee.Image().byte()
               .paint(ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([w, s, e, n]))]), 1, 2)
               .visualize(palette=["#ffeb3b"]))

    out: Dict = {"labels": {}, "scenes": {}}
    for key, (start, end) in image_windows(today).items():
        rgb, count = _composite(ee, start, end, region)
        scenes = int(call_with_retry(count.getInfo))
        out["scenes"][key] = scenes
        out["labels"][key] = f"{start:%b %Y} – {end:%b %Y}"
        if scenes == 0:
            out[key] = None            # no usable scenes: say so instead of showing a blank tile
            continue
        vis = rgb.visualize(min=0, max=3000, gamma=1.2)
        composite = ee.ImageCollection([vis, outline]).mosaic()
        out[key] = call_with_retry(
            lambda img=composite: img.getThumbURL(
                {"region": region, "dimensions": THUMB_PX, "format": "jpg"}))
    return out
