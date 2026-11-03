"""
TerraVision AI — headless scan pipeline (no Streamlit).

One code path used by both the web app and scheduled_scan.py:

    satellite rasters (rolling latest window, cached)
      -> change detection + classification            (detect.py)
      -> cross-checks from other datasets             (enrichment.py)
      -> tracking against the previous scan           (tracking.py)
      -> evidence-adjusted confidence                 (evidence.py)
      -> saved to SQLite                              (store.py)
"""

from __future__ import annotations

import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import time

import evidence
import store
import tracking
from reliability import call_with_retry, get_logger
from regions import REGION_BOUNDS, region_slug

VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
BASELINE_RANGE = ("2019-01-01", "2023-12-31")
CURRENT_MONTHS = 2                       # average of the latest N monthly composites
PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "data_cache"

Progress = Callable[[str], None]


# ── rolling window ───────────────────────────────────────────────────────────

def _add_months(d: date, n: int) -> date:
    m = d.year * 12 + (d.month - 1) + n
    return date(m // 12, m % 12 + 1, 1)


def rolling_window(latest_image_date: date, months: int = CURRENT_MONTHS) -> Tuple[date, date, str]:
    """
    Window covering the newest `months` monthly composites.
    Returns (start, end_exclusive, label) where label is the newest month, e.g. "2026-08".
    """
    newest = date(latest_image_date.year, latest_image_date.month, 1)
    start = _add_months(newest, -(months - 1))
    end = _add_months(newest, 1)
    return start, end, f"{newest.year}-{newest.month:02d}"


def latest_viirs_date(ee, bounds: Tuple[float, float, float, float]) -> date:
    """Date of the newest VIIRS monthly composite over the region (asks Earth Engine)."""
    geom = ee.Geometry.Rectangle(list(bounds))
    ms = call_with_retry(
        lambda: ee.ImageCollection(VIIRS_COLLECTION).filterBounds(geom)
        .aggregate_max("system:time_start").getInfo())
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


# ── satellite rasters ────────────────────────────────────────────────────────

def _download(ee, image, geom, out_path: Path) -> None:
    def fetch() -> None:
        url = image.getDownloadURL({"bands": ["avg_rad", "cf_cvg"], "region": geom, "scale": 500, "format": "GEO_TIFF"})
        tmp = out_path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        tmp.replace(out_path)

    call_with_retry(fetch)


def get_rasters(region: str, refresh: bool = False, progress: Progress = lambda m: None
                ) -> Tuple[str, str, str, List[str]]:
    """
    Returns (baseline_path, current_path, period_label, notes).

    Preference order for the *current* raster:
      1. cached download for the newest available month (data_cache/)
      2. fresh download from Earth Engine
      3. offline fallback: newest cached download, or the old hand-exported
         viirs_current_<region>.tif — flagged as possibly out of date.
    `refresh=True` skips (1) and downloads again.
    Raises RuntimeError if no raster can be found at all.
    """
    slug = region_slug(region)
    notes: List[str] = []
    CACHE_DIR.mkdir(exist_ok=True)

    legacy_baseline = [PROJECT_DIR / f"viirs_baseline_{slug}.tif"]
    if region == "Western Ghats, India":
        legacy_baseline.append(PROJECT_DIR / "viirs_baseline.tif")
    # Quality-masked baseline (radiance + cloud-free count) is built by this app; a hand-exported
    # single-band file is only an offline fallback because it cannot be quality-checked.
    baseline = CACHE_DIR / f"viirs_baseline_{slug}_q2.tif"
    legacy = next((p for p in legacy_baseline if p.exists()), None)

    try:
        import ee
        from gee_gateway import connect_gee

        connect_gee()
        bounds = REGION_BOUNDS[region]
        geom = ee.Geometry.Rectangle(list(bounds))

        progress("Checking for the newest VIIRS month…")
        start, end, label = rolling_window(latest_viirs_date(ee, bounds))
        current = CACHE_DIR / f"viirs_current_{slug}_{label}_q2.tif"
        collection = ee.ImageCollection(VIIRS_COLLECTION).filterBounds(geom).select(["avg_rad", "cf_cvg"])

        if not baseline.exists() or refresh:
            progress("Downloading 5-year baseline from Earth Engine (1–2 min)…")
            _download(ee, collection.filterDate(*BASELINE_RANGE).median(), geom, baseline)
        if not current.exists() or refresh:
            progress(f"Downloading latest satellite data ({start:%b %Y} – {end:%b %Y}, 1–2 min)…")
            _download(ee, collection.filterDate(start.isoformat(), end.isoformat()).mean(), geom, current)
        else:
            progress(f"Using cached satellite data for {label}")
        return str(baseline), str(current), label, notes

    except Exception as e:
        # Offline / auth problem: fall back to whatever real data is already on disk.
        cached = sorted(CACHE_DIR.glob(f"viirs_current_{slug}_*_q2.tif"))
        if cached and baseline.exists():
            label = cached[-1].stem[:-len("_q2")].rsplit("_", 1)[-1]
            notes.append(f"Earth Engine unreachable ({str(e)[:120]}); used cached data for {label}.")
            return str(baseline), str(cached[-1]), label, notes

        legacy_current = [PROJECT_DIR / f"viirs_current_{slug}.tif"]
        if region == "Western Ghats, India":
            legacy_current.append(PROJECT_DIR / "viirs_current.tif")
        cur = next((p for p in legacy_current if p.exists()), None)
        if cur and legacy:
            baseline = legacy
            notes.append(f"Earth Engine unreachable ({str(e)[:120]}); used a previously exported "
                         f"local file, which may be out of date.")
            return str(baseline), str(cur), "local-export", notes

        raise RuntimeError(f"No satellite data available for {region}: {e}") from e


def get_window_rasters(region: str, start: date, end: date, label: str,
                       progress: Progress = lambda m: None) -> Tuple[str, str]:
    """
    (baseline_path, window_path) for an arbitrary [start, end) window, quality-masked and cached.
    Used to build training data from past months; needs Earth Engine.
    """
    import ee
    from gee_gateway import connect_gee

    slug = region_slug(region)
    CACHE_DIR.mkdir(exist_ok=True)
    connect_gee()
    geom = ee.Geometry.Rectangle(list(REGION_BOUNDS[region]))
    collection = ee.ImageCollection(VIIRS_COLLECTION).filterBounds(geom).select(["avg_rad", "cf_cvg"])
    baseline = CACHE_DIR / f"viirs_baseline_{slug}_q2.tif"
    window = CACHE_DIR / f"viirs_win_{slug}_{label}_q2.tif"
    if not baseline.exists():
        progress("Downloading 5-year baseline…")
        _download(ee, collection.filterDate(*BASELINE_RANGE).median(), geom, baseline)
    if not window.exists():
        progress(f"Downloading VIIRS {start:%b %Y} – {end:%b %Y}…")
        _download(ee, collection.filterDate(start.isoformat(), end.isoformat()).mean(), geom, window)
    return str(baseline), str(window)


# ── the scan ─────────────────────────────────────────────────────────────────

def _run_scan_impl(
    region: str,
    threshold: float = 5.0,
    min_cluster: int = 3,
    enrich: bool = True,
    refresh: bool = False,
    save: bool = True,
    progress: Progress = lambda m: None,
    store_path: Optional[Path] = None,
    regional_correction: bool = True,
) -> Dict:
    """Run a complete scan of one region and return the scan dict (see store.save_scan)."""
    from detect import DetectionParams, detect_encroachments

    baseline, current, period, notes = get_rasters(region, refresh=refresh, progress=progress)

    progress("Detecting light clusters and classifying them…")
    alerts = detect_encroachments(baseline, current,
                                  params=DetectionParams(delta_min_flag=threshold, regional_correction=regional_correction),
                                  verbose=False)
    alerts = [a for a in alerts if a["cluster_size"] >= min_cluster]

    scanned_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    for a in alerts:
        a["region"] = region
        a["detected_at"] = scanned_at
        a["period"] = period

    context_ok: List[str] = []
    if enrich and alerts:
        try:
            from enrichment import add_context
            _, context_ok, ctx_notes = add_context(alerts, progress=progress)
            notes += ctx_notes
        except Exception as e:
            notes.append(f"Cross-checks skipped: {type(e).__name__}: {str(e)[:160]}")
    elif enrich:
        context_ok = []

    # Real region footprint: official file or designated protected areas, tested locally
    boundary_info = ""
    if alerts:
        try:
            import boundaries
            if enrich:
                bnd, bnote = boundaries.ensure_boundary(region, progress=progress, refresh=refresh)
                if bnote:
                    notes.append(bnote)
            else:
                bnd = boundaries.load_boundary(region)
            if bnd:
                boundaries.annotate_alerts(alerts, bnd, fill_protected_area="protected_area" not in context_ok)
                boundary_info = boundaries.describe(bnd)
        except Exception as e:
            notes.append(f"Boundary check skipped: {type(e).__name__}: {str(e)[:160]}")

    progress("Comparing with the previous scan…")
    prev = store.previous_scan(region, store_path)
    prev_alerts = prev["alerts"] if prev else []
    comparable = (bool(prev) and prev.get("threshold") == threshold and prev.get("min_cluster") == min_cluster
                  and prev.get("regional_correction", False) == regional_correction)
    _, resolved = tracking.apply_tracking(
        region, alerts, prev_alerts, scanned_at, period,
        previous_period=prev.get("period") if prev else None, comparable=comparable,
    )
    if prev and not comparable:
        notes.append("Previous scan used different detection settings, so 'resolved' alerts are not reported.")

    evidence.apply_evidence(alerts)

    scan = {
        "region": region, "scanned_at": scanned_at, "period": period,
        "threshold": threshold, "min_cluster": min_cluster, "regional_correction": regional_correction,
        "context_ok": context_ok, "notes": notes, "resolved": resolved, "alerts": alerts,
        "had_previous": bool(prev), "boundary": boundary_info,
    }
    if save:
        scan["scan_id"] = store.save_scan(scan, store_path)
    return scan


def run_scan(
    region: str,
    threshold: float = 5.0,
    min_cluster: int = 3,
    enrich: bool = True,
    refresh: bool = False,
    save: bool = True,
    progress: Progress = lambda m: None,
    store_path: Optional[Path] = None,
    source: str = "app",
    **kwargs,
) -> Dict:
    """
    Run a complete scan of one region (see _run_scan_impl) and record the attempt —
    success or failure, duration, layers that worked — in the run history.
    """
    log = get_logger()
    started = datetime.now()
    t0 = time.monotonic()
    log.info("Scan started: %s (threshold %.1f, min cluster %d, cross-checks %s)",
             region, threshold, min_cluster, enrich)
    try:
        scan = _run_scan_impl(region, threshold=threshold, min_cluster=min_cluster, enrich=enrich,
                              refresh=refresh, save=save, progress=progress, store_path=store_path, **kwargs)
    except Exception as e:
        log.exception("Scan failed: %s", region)
        if save:
            try:
                store.log_run(region, started.strftime("%Y-%m-%d %H:%M"), time.monotonic() - t0, "failed",
                              f"{type(e).__name__}: {e}", source=source, path=store_path)
            except Exception:
                log.exception("Could not record the failed run")
        raise
    duration = time.monotonic() - t0
    log.info("Scan finished: %s — %d alerts in %.0fs (layers ok: %s)", region, len(scan["alerts"]),
             duration, ", ".join(scan["context_ok"]) or "none")
    for note in scan.get("notes", []):
        log.warning("%s: %s", region, note)
    if save:
        try:
            store.log_run(region, started.strftime("%Y-%m-%d %H:%M"), duration, "ok", "",
                          len(scan["alerts"]), scan.get("period") or "", scan["context_ok"],
                          source=source, path=store_path)
        except Exception:
            log.exception("Could not record the run")
    return scan
