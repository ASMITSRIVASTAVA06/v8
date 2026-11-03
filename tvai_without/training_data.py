"""
TerraVision AI — build a real, independently labelled training set.

The classifier only ever sees the SHAPE and BRIGHTNESS of a night-light cluster. The labels it
learns from come from other satellites that it never sees:

  active fire (NASA FIRMS) + tree-covered land   -> natural_wildfire
  active fire (FIRMS)      + cropland            -> agricultural_burn
  built-up land >= 30 %                          -> anomaly            (town / industry, not forest)
  no fire, tree loss or NDVI drop, in forest or a
  protected area, little built-up land           -> illegal_encroachment
  anything ambiguous                             -> dropped, never guessed

This is weak supervision, not ground truth: expect some label noise, and review real alerts in the
Alerts section — your verdicts are merged in with extra weight when you train.

  python training_data.py --regions all --windows 6      # needs Earth Engine; resumable
  python train_real.py                                   # then train on the result
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from regions import REGION_BOUNDS, region_slug

DATA_DIR = Path(__file__).parent / "training"
DATASET_CSV = DATA_DIR / "real_dataset.csv"
DONE_JSON = DATA_DIR / "done_windows.json"

FEATURE_NAMES = ["delta_mean", "delta_max", "delta_std", "cluster_size", "aspect_ratio", "compactness",
                 "convexity", "edge_ratio", "spread_rate", "interior_uniformity", "bbox_area"]
COLUMNS = ["region", "window", "lat", "lon"] + FEATURE_NAMES + ["label", "weight", "source"]

TRAIN_START = date(2024, 1, 1)      # the baseline covers 2019-2023, so training windows come strictly after it
WINDOW_MONTHS = 2

URBAN = 0.30
BUILTUP_MAX_FOR_ENCROACH = 0.15
FOREST_CONTEXT = 0.30
CROP_CONTEXT = 0.30
LOSS_MIN = 0.02
NDVI_DROP = -0.08


def label_cluster(a: Dict) -> Tuple[Optional[str], float]:
    """
    (label, weight) from independent evidence, or (None, 0) when the evidence does not decide.
    Needs the cross-check fields; any missing key signal means the cluster is dropped.
    """
    builtup, fire = a.get("builtup_fraction"), a.get("fire_detected")
    if builtup is None or fire is None:
        return None, 0.0
    tree = a.get("tree_context")
    tree = a.get("tree_fraction") if tree is None else tree
    crop = a.get("crop_context") or 0.0
    in_pa = a.get("in_protected_area")
    loss, ndvi = a.get("tree_loss_fraction"), a.get("ndvi_change")

    if builtup >= URBAN:
        return "anomaly", 1.0
    if fire:
        if crop >= CROP_CONTEXT and (tree or 0) < FOREST_CONTEXT:
            return "agricultural_burn", 1.0
        if (tree or 0) >= FOREST_CONTEXT and crop < CROP_CONTEXT:
            return "natural_wildfire", 1.0
        return None, 0.0                                   # fire in mixed land: ambiguous
    if builtup < BUILTUP_MAX_FOR_ENCROACH and (loss is not None or ndvi is not None):
        cleared = (loss is not None and loss >= LOSS_MIN) or (ndvi is not None and ndvi <= NDVI_DROP)
        in_forest = in_pa is True or (tree or 0) >= FOREST_CONTEXT
        if cleared and in_forest:
            return "illegal_encroachment", 1.0
        if not cleared and not in_forest and in_pa is False and builtup < 0.05:
            return "anomaly", 0.5                          # bright, but nowhere near forest and nothing changed
    return None, 0.0


def training_windows(today: date, n: int, months: int = WINDOW_MONTHS) -> List[Tuple[date, date, str]]:
    """The newest `n` complete, non-overlapping windows that start on or after TRAIN_START."""
    def add(d: date, k: int) -> date:
        m = d.year * 12 + d.month - 1 + k
        return date(m // 12, m % 12 + 1, 1)

    end = date(today.year, today.month, 1)              # exclude the current, incomplete month
    out = []
    while len(out) < n:
        start = add(end, -months)
        if start < TRAIN_START:
            break
        out.append((start, end, f"{start:%Y-%m}_{months}m"))
        end = start
    return out[::-1]


def _load_done() -> Dict[str, int]:
    try:
        return json.loads(DONE_JSON.read_text())
    except (OSError, ValueError):
        return {}


def _append_rows(rows: List[Dict], path: Path = DATASET_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def rows_from_scan(region: str, window: str, alerts: List[Dict], feats: List) -> List[Dict]:
    """Labelled CSV rows for one region and window (clusters without a decisive label are dropped)."""
    rows = []
    for a, f in zip(alerts, feats):
        label, weight = label_cluster(a)
        if label is None:
            continue
        row = {"region": region, "window": window, "lat": round(a["lat"], 5), "lon": round(a["lon"], 5),
               "label": label, "weight": weight, "source": "auto"}
        for name in FEATURE_NAMES:
            row[name] = float(getattr(f, name))
        rows.append(row)
    return rows


def build_dataset(regions: List[str], n_windows: int, progress: Callable[[str], None] = print,
                  today: Optional[date] = None, max_per_window: int = 3000) -> int:
    """Download, detect, cross-check and label; returns the number of rows added. Resumable."""
    import numpy as np

    from detect import DetectionParams, extract_clusters, read_delta
    from enrichment import add_context
    from pipeline import get_window_rasters

    params = DetectionParams(delta_min_flag=3.0, regional_correction=True)
    done, added = _load_done(), 0
    for region in regions:
        for start, end, label in training_windows(today or date.today(), n_windows):
            key = f"{region_slug(region)}|{label}"
            if key in done:
                continue
            progress(f"[{region}] {start:%b %Y} – {end:%b %Y}")
            try:
                base, win = get_window_rasters(region, start, end, label, progress)
                delta, transform, _ = read_delta(base, win, params)
                alerts, feats, _ = extract_clusters(
                    delta, lambda r, c: tuple(transform * (c + 0.5, r + 0.5)), params, classify=False)
                if len(alerts) > max_per_window:           # keep the strongest; labelling cost grows with count
                    order = np.argsort([-a["delta_max"] for a in alerts])[:max_per_window]
                    alerts, feats = [alerts[i] for i in order], [feats[i] for i in order]
                if alerts:
                    add_context(alerts, progress=progress, today=end - timedelta(days=1))
                rows = rows_from_scan(region, label, alerts, feats)
                _append_rows(rows)
                done[key] = len(rows)
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                DONE_JSON.write_text(json.dumps(done, indent=1))
                added += len(rows)
                progress(f"   kept {len(rows)} labelled of {len(alerts)} clusters")
            except Exception as e:                           # one bad window must not lose the others
                progress(f"   skipped: {type(e).__name__}: {str(e)[:160]}")
    return added


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build the real, independently labelled training set.")
    p.add_argument("--regions", default="all", help="'all' or names separated by ';'")
    p.add_argument("--windows", type=int, default=6, help="how many 2-month windows per region (newest first)")
    args = p.parse_args(argv)
    from gee_gateway import connect_gee
    connect_gee()
    regions = list(REGION_BOUNDS) if args.regions == "all" else [r.strip() for r in args.regions.split(";")]
    unknown = [r for r in regions if r not in REGION_BOUNDS]
    if unknown:
        raise SystemExit(f"Unknown regions: {unknown}")
    n = build_dataset(regions, args.windows)
    print(f"\nAdded {n} labelled clusters -> {DATASET_CSV}\nNext: python train_real.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
