"""
TerraVision AI — Detection Pipeline (ML-enhanced)
==================================================
Drop-in replacement for the original rule-based detect.py.

What changed
────────────
classify_cluster() now calls the Random Forest model from model.py
instead of four hand-crafted scoring functions.

All other pipeline stages (raster loading, delta computation, clustering,
feature extraction, severity mapping, output schema) are UNCHANGED, so
app.py works exactly as before with higher accuracy.

The rule-engine scoring functions are preserved as fallbacks in case
the model file is absent (e.g. first run before model.py is executed).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage          # rasterio is imported lazily, only where a file is read

import regional as _regional


# =============================================================================
# 1. TUNEABLE PARAMETER SET  (unchanged — used by feature extraction and app.py)
# =============================================================================

@dataclass
class DetectionParams:
    """All tuneable thresholds in one place. Load from JSON with from_json()."""

    delta_min_flag: float = 2.5
    delta_agri_max: float = 9.0
    delta_wildfire_min: float = 14.0
    delta_encroach_max: float = 13.0
    delta_anomaly_min: float = 20.0

    size_min_valid: int = 3
    size_encroach_max: int = 18
    size_agri_min: int = 25
    size_wildfire_min: int = 15
    size_anomaly_max: int = 20

    aspect_agri_min: float = 1.8
    compactness_wildfire_max: float = 0.65
    compactness_agri_min: float = 0.55
    convexity_encroach_max: float = 0.80
    uniformity_agri_min: float = 0.55

    spread_rate_wildfire_min: float = 0.40

    w_radiance: float = 0.35
    w_size: float = 0.25
    w_shape: float = 0.25
    w_context: float = 0.15

    # Remove the district-wide brightening trend before looking for clusters (see regional.py)
    regional_correction: bool = True
    regional_block: int = 16

    # Data quality: ignore pixels seen cloud-free fewer than this many times in the composite
    # (VIIRS band 2, cf_cvg). Dim, cloudy or snowy pixels give unreliable radiance.
    min_cloud_free_obs: float = 2.0

    @classmethod
    def from_json(cls, path: str) -> "DetectionParams":
        with open(path) as f:
            return cls(**json.load(f))


DEFAULT_PARAMS = DetectionParams()

# Feature order must match exactly what model.py was trained on
FEATURE_NAMES = [
    "delta_mean", "delta_max", "delta_std", "cluster_size",
    "aspect_ratio", "compactness", "convexity", "edge_ratio",
    "spread_rate", "interior_uniformity", "bbox_area",
]

CLASS_NAMES = ["natural_wildfire", "agricultural_burn",
               "illegal_encroachment", "anomaly"]

SEVERITY_MAP = {
    "natural_wildfire":     "critical",
    "agricultural_burn":    "warning",
    "illegal_encroachment": "high",
    "anomaly":              "medium",
    "unclassified":         "low",
}

# Lazy-loaded model singleton — loaded once and reused for every cluster
_rf_model = None


def _retrain_after_version_change():
    """Rebuild a stale classifier: from the real labelled dataset when there is one, else synthetic."""
    try:
        import train_real
        from training_data import DATASET_CSV
        if DATASET_CSV.exists():
            d = train_real.load_dataset(DATASET_CSV)
            model, report = train_real.train_from_data(d, force=True, log=lambda *_: None)
            if model is not None:
                return model
    except Exception as e:
        print(f"Could not retrain on real data ({e}); falling back to synthetic examples.")
    from model import train as _train
    return _train()


def _get_rf_model():
    """
    Load the Random Forest model from model.py.
    Returns the model if available, None if not yet trained.
    Prints a one-time message indicating which classifier is being used.
    """
    global _rf_model
    if _rf_model is not None:
        return _rf_model

    model_path = Path("terravision_rf.joblib")
    if model_path.exists():
        try:
            import warnings
            import joblib
            try:
                from sklearn.exceptions import InconsistentVersionWarning
            except ImportError:                      # very old scikit-learn: no such warning exists
                InconsistentVersionWarning = None
            with warnings.catch_warnings():
                if InconsistentVersionWarning is not None:
                    warnings.simplefilter("error", InconsistentVersionWarning)
                _rf_model = joblib.load(model_path)
            return _rf_model
        except Exception as e:
            # A model saved by a different scikit-learn version can give silently wrong results,
            # so rebuild it with the installed version instead of using it.
            if type(e).__name__ == "InconsistentVersionWarning":
                print("Saved classifier was built with a different scikit-learn version; "
                      "retraining it now (one-off, about a minute)...")
                try:
                    _rf_model = _retrain_after_version_change()
                    return _rf_model
                except Exception as retrain_error:
                    print(f"Retraining failed ({retrain_error}); using the rule-based classifier.")
            return None

    # Model not trained yet — try to auto-train
    try:
        from model import train as _train
        _rf_model = _train()
        return _rf_model
    except Exception:
        return None


# =============================================================================
# 2. CONNECTED-COMPONENT CLUSTERING  (unchanged)
# =============================================================================

def find_clusters(mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Label 8-connected components; return (label_map, {label: size}).

    Vectorized with scipy.ndimage.label (same 8-connectivity as the
    original pure-Python flood-fill, just implemented in C). The old
    row-by-row Python loop was fine for small prototype rasters but
    doesn't scale to full-resolution GEE downloads, which can be tens
    of millions of pixels for large regions (e.g. Amazon Basin).
    """
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity (incl. diagonals)
    labels, num_features = ndimage.label(mask, structure=structure)
    labels = labels.astype(np.int32)

    if num_features == 0:
        return labels, {}

    counts = np.bincount(labels.ravel())
    sizes: Dict[int, int] = {
        lbl: int(counts[lbl]) for lbl in range(1, num_features + 1)
    }
    return labels, sizes


# =============================================================================
# 3. FEATURE EXTRACTION  (unchanged)
# =============================================================================

@dataclass
class ClusterFeatures:
    label: int
    centroid_row: int
    centroid_col: int
    cluster_size: int
    delta_mean: float
    delta_max: float
    delta_std: float
    aspect_ratio: float
    compactness: float
    convexity: float
    edge_ratio: float
    spread_rate: float
    interior_uniformity: float
    bbox_area: int


def _extract_features(label_map: np.ndarray, lbl: int, delta: np.ndarray,
                       row_offset: int = 0, col_offset: int = 0) -> ClusterFeatures:
    """
    label_map / delta are expected to already be cropped to this cluster's
    bounding box (see detect_encroachments, which uses
    scipy.ndimage.find_objects so it doesn't rescan the full raster once
    per cluster — that used to be O(num_clusters * raster_size), which is
    fine for a small test raster but grinds to a halt on a real multi-
    million-pixel GEE download). row_offset/col_offset translate the local
    (cropped) coordinates back to the full raster's coordinate space.
    """
    pixels  = np.argwhere(label_map == lbl)
    rows_px = pixels[:, 0]
    cols_px = pixels[:, 1]
    size    = len(pixels)

    cr = int(rows_px.mean()) + row_offset
    cc = int(cols_px.mean()) + col_offset

    vals   = delta[rows_px, cols_px]
    d_mean = float(vals.mean())
    d_max  = float(vals.max())
    d_std  = float(vals.std()) if size > 1 else 0.0

    r_min, r_max = int(rows_px.min()), int(rows_px.max())
    c_min, c_max = int(cols_px.min()), int(cols_px.max())
    bb_h = max(r_max - r_min + 1, 1)
    bb_w = max(c_max - c_min + 1, 1)
    bbox_area = bb_h * bb_w

    aspect_ratio = max(bb_w / bb_h, bb_h / bb_w)
    spread_rate  = size / bbox_area

    r_off, c_off = r_min, c_min
    local = np.zeros((bb_h, bb_w), dtype=bool)
    local[rows_px - r_off, cols_px - c_off] = True
    perimeter = 0
    for pr, pc in zip(rows_px - r_off, cols_px - c_off):
        is_boundary = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr2, nc2 = pr + dr, pc + dc
                if not (0 <= nr2 < bb_h and 0 <= nc2 < bb_w) or not local[nr2, nc2]:
                    is_boundary = True
                    break
            if is_boundary:
                break
        if is_boundary:
            perimeter += 1

    perimeter   = max(perimeter, 1)
    edge_ratio  = perimeter / size
    compactness = min(1.0, (4 * math.pi * size) / (perimeter ** 2))

    try:
        from scipy.spatial import ConvexHull
        hull      = ConvexHull(pixels)
        hull_area = hull.volume
    except Exception:
        hull_area = float(bbox_area)
    convexity = min(1.0, size / max(hull_area, 1))

    if d_mean > 0:
        interior_uniformity = max(0.0, 1.0 - (d_std / d_mean))
    else:
        interior_uniformity = 0.0

    return ClusterFeatures(
        label=lbl, centroid_row=cr, centroid_col=cc,
        cluster_size=size, delta_mean=d_mean, delta_max=d_max, delta_std=d_std,
        aspect_ratio=aspect_ratio, compactness=compactness, convexity=convexity,
        edge_ratio=edge_ratio, spread_rate=spread_rate,
        interior_uniformity=interior_uniformity, bbox_area=bbox_area,
    )


# =============================================================================
# 4. ML-POWERED CLASSIFIER
# =============================================================================

def _class_names(model) -> List[str]:
    """Class names in the order of the model's probability columns (ints 0..3 or the names themselves)."""
    out = []
    for c in getattr(model, "classes_", range(len(CLASS_NAMES))):
        out.append(c if isinstance(c, str) else CLASS_NAMES[int(c)])
    return out


def classify_cluster(
    f: ClusterFeatures,
    p: DetectionParams = DEFAULT_PARAMS,
) -> Tuple[str, str, int, Dict[str, float]]:
    """
    Classify a cluster using the Random Forest model.
    Falls back to the rule engine if the model is unavailable.
    """
    if f.cluster_size < p.size_min_valid:
        return "unclassified", "low", 10, {}

    model = _get_rf_model()

    if model is not None:
        # ML path
        x = np.array([[
            f.delta_mean, f.delta_max, f.delta_std, float(f.cluster_size),
            f.aspect_ratio, f.compactness, f.convexity, f.edge_ratio,
            f.spread_rate, f.interior_uniformity, float(f.bbox_area),
        ]], dtype=np.float32)

        proba    = model.predict_proba(x)[0]
        names    = _class_names(model)
        pred_idx = int(proba.argmax())
        top_prob = float(proba[pred_idx])

        # Confidence: scale probability to 20-97 range
        # Apply a margin penalty when top two scores are close (genuine ambiguity)
        sorted_p = np.sort(proba)[::-1]
        margin   = sorted_p[0] - sorted_p[1]
        base_conf      = int(min(97, max(20, top_prob * 100)))
        margin_penalty = int(max(0, (0.15 - margin) / 0.15 * 12))
        confidence     = max(15, base_conf - margin_penalty)

        # Low-confidence predictions returned as unclassified
        category = names[pred_idx] if top_prob >= 0.35 else "unclassified"
        severity = SEVERITY_MAP.get(category, "low")
        scores   = {names[i]: round(float(pr), 4) for i, pr in enumerate(proba)}
        return category, severity, confidence, scores

    # Rule-based fallback
    return _rule_classify(f, p)


# =============================================================================
# 4b. RULE ENGINE FALLBACK  (used only when model file is missing)
# =============================================================================

def _rule_classify(f: ClusterFeatures,
                   p: DetectionParams) -> Tuple[str, str, int, Dict[str, float]]:
    scores = {
        "natural_wildfire":     _score_wildfire(f, p),
        "agricultural_burn":    _score_agri(f, p),
        "illegal_encroachment": _score_encroach(f, p),
        "anomaly":              _score_anomaly(f, p),
    }
    best_cat   = max(scores, key=lambda k: scores[k])
    best_score = scores[best_cat]
    sorted_s   = sorted(scores.values(), reverse=True)
    margin     = sorted_s[0] - sorted_s[1] if len(sorted_s) > 1 else sorted_s[0]
    base_conf  = int(min(97, max(20, best_score * 120)))
    confidence = max(15, base_conf - int(max(0, (0.15 - margin) / 0.15 * 20)))
    if best_score < 0.20:
        return "unclassified", "low", confidence, scores
    return best_cat, SEVERITY_MAP.get(best_cat, "low"), confidence, scores


def _score_wildfire(f: ClusterFeatures, p: DetectionParams) -> float:
    r = min(1.0, max(0.0, (f.delta_max - p.delta_wildfire_min) / (p.delta_wildfire_min * 1.5)))
    s = min(1.0, max(0.0, (f.cluster_size - p.size_wildfire_min) / (p.size_wildfire_min * 4)))
    sh = (1.0 - f.compactness)*0.5 + min(1.0, f.spread_rate/p.spread_rate_wildfire_min)*0.3 + (1.0-f.convexity)*0.2
    cv = f.delta_std / max(f.delta_mean, 0.001)
    sc = (p.w_radiance*r + p.w_size*s + p.w_shape*sh + p.w_context*min(1.0, cv/0.5))
    if f.delta_max < p.delta_wildfire_min * 0.7: sc *= 0.3
    if f.cluster_size < p.size_wildfire_min: sc *= 0.4
    return round(sc, 4)


def _score_agri(f: ClusterFeatures, p: DetectionParams) -> float:
    if f.delta_mean <= p.delta_min_flag: r = 0.0
    elif f.delta_mean <= p.delta_agri_max: r = (f.delta_mean - p.delta_min_flag) / (p.delta_agri_max - p.delta_min_flag)
    else: r = max(0.0, 1.0 - (f.delta_mean - p.delta_agri_max) / p.delta_agri_max)
    s  = min(1.0, max(0.0, (f.cluster_size - p.size_agri_min) / (p.size_agri_min * 3)))
    sh = min(1.0, (f.aspect_ratio-1.0)/(p.aspect_agri_min-1.0+0.01))*0.4 + min(1.0, f.compactness/p.compactness_agri_min)*0.3 + f.interior_uniformity*0.3
    sc = (p.w_radiance*r + p.w_size*s + p.w_shape*sh + p.w_context*f.interior_uniformity)
    if f.delta_max > p.delta_wildfire_min: sc *= 0.5
    if f.cluster_size < p.size_agri_min: sc *= 0.3
    return round(sc, 4)


def _score_encroach(f: ClusterFeatures, p: DetectionParams) -> float:
    if f.delta_mean < p.delta_min_flag: r = 0.0
    elif f.delta_mean <= p.delta_encroach_max: r = 0.6 + 0.4*((f.delta_mean-p.delta_min_flag)/(p.delta_encroach_max-p.delta_min_flag))
    else: r = max(0.0, 1.0-(f.delta_mean-p.delta_encroach_max)/p.delta_encroach_max)
    s = max(0.0, 1.0-(f.cluster_size-1)/(p.size_encroach_max*1.5)) if f.cluster_size<=p.size_encroach_max else max(0.0, 1.0-(f.cluster_size-p.size_encroach_max)/(p.size_agri_min-p.size_encroach_max+1))
    sh = f.edge_ratio*0.5 + (1.0-abs(f.spread_rate-0.5))*0.5
    sc = (p.w_radiance*r + p.w_size*s + p.w_shape*sh + p.w_context*0.5)
    if f.cluster_size > p.size_agri_min: sc *= 0.3
    if f.delta_max > p.delta_wildfire_min: sc *= 0.4
    return round(sc, 4)


def _score_anomaly(f: ClusterFeatures, p: DetectionParams) -> float:
    r  = min(1.0, max(0.0, (f.delta_max - p.delta_anomaly_min) / p.delta_anomaly_min))
    s  = max(0.0, 1.0 - f.cluster_size / (p.size_anomaly_max * 2))
    sh = f.compactness * 0.5 + f.interior_uniformity * 0.5
    return round(p.w_radiance*r + p.w_size*s + p.w_shape*sh + p.w_context*min(1.0, f.delta_max/(p.delta_anomaly_min*2)), 4)


# =============================================================================
# 5. MAIN DETECTION PIPELINE  (unchanged interface)
# =============================================================================

def read_delta(baseline_path: str, current_path: str, params: DetectionParams = DEFAULT_PARAMS):
    """
    Radiance change (current - baseline) with unreliable pixels set to NaN, plus the affine transform.

    Band 1 is average radiance. If a file also has band 2 (cloud-free observation count) pixels
    with fewer than `min_cloud_free_obs` clear observations are masked out; older single-band files
    are used as they are. Returns (delta, transform, masked_fraction).
    """
    import rasterio

    def load(path):
        with rasterio.open(path) as src:
            rad = src.read(1).astype(float)
            cf = src.read(2).astype(float) if src.count >= 2 else None
            return rad, cf, src.transform

    base, base_cf, transform = load(baseline_path)
    cur, cur_cf, _ = load(current_path)
    delta = cur - base
    bad = np.zeros(delta.shape, dtype=bool)
    for cf in (base_cf, cur_cf):
        if cf is not None and params.min_cloud_free_obs > 0:
            bad |= ~(cf >= params.min_cloud_free_obs)
    masked = float(bad.mean()) if bad.size else 0.0
    delta[bad] = np.nan
    return delta, transform, masked


def extract_clusters(delta: np.ndarray, xy, params: DetectionParams = DEFAULT_PARAMS, classify: bool = True):
    """
    Find, measure and classify bright clusters in a delta raster.
    `xy(row, col) -> (lon, lat)` converts pixel to map coordinates.
    With classify=False no model is loaded (training-data building).
    Returns (alerts, features_list, suspicious_pixel_count); features_list is aligned with alerts.
    """
    if params.regional_correction:
        delta = _regional.correct_delta(delta, params.regional_block)
    suspicious_mask = (delta > params.delta_min_flag) & np.isfinite(delta)

    labels, cluster_sizes = find_clusters(suspicious_mask)
    unique_labels = [lbl for lbl, sz in cluster_sizes.items() if sz >= params.size_min_valid]

    # One O(raster_size) pass to get every cluster's bounding box
    bboxes = ndimage.find_objects(labels)

    alerts, feats = [], []
    for lbl in unique_labels:
        bbox = bboxes[lbl - 1]
        if bbox is None:
            continue
        row_slice, col_slice = bbox
        features = _extract_features(
            labels[row_slice, col_slice], lbl, delta[row_slice, col_slice],
            row_offset=row_slice.start, col_offset=col_slice.start,
        )
        if classify:
            category, severity, confidence, raw_scores = classify_cluster(features, params)
        else:                                   # used to build training data: no model involved
            category, severity, confidence, raw_scores = "unclassified", "low", 0, {}
        lon, lat = xy(features.centroid_row, features.centroid_col)
        alerts.append({
            "lat":          float(lat),
            "lon":          float(lon),
            "delta_mean":   round(features.delta_mean, 3),
            "delta_max":    round(features.delta_max, 3),
            "delta_std":    round(features.delta_std, 3),
            "bbox_area":    int(features.bbox_area),
            "severity":     severity,
            "category":     category,
            "cluster_size": features.cluster_size,
            "confidence":   confidence,
            "shape": {
                "aspect_ratio":        round(features.aspect_ratio, 3),
                "compactness":         round(features.compactness, 3),
                "convexity":           round(features.convexity, 3),
                "edge_ratio":          round(features.edge_ratio, 3),
                "spread_rate":         round(features.spread_rate, 3),
                "interior_uniformity": round(features.interior_uniformity, 3),
            },
            "scores": {k: round(v, 4) for k, v in raw_scores.items()},
        })
        feats.append(features)

    severity_rank = {"critical": 0, "high": 1, "warning": 2, "medium": 3, "low": 4}
    order = sorted(range(len(alerts)),
                   key=lambda i: (severity_rank.get(alerts[i]["severity"], 99), -alerts[i]["confidence"]))
    return [alerts[i] for i in order], [feats[i] for i in order], int(suspicious_mask.sum())


def detect_encroachments(
    baseline_path: str,
    current_path:  str,
    params: DetectionParams = DEFAULT_PARAMS,
    params_json: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict]:
    if params_json:
        params = DetectionParams.from_json(params_json)

    delta, transform, masked = read_delta(baseline_path, current_path, params)
    alerts, _, n_suspicious = extract_clusters(
        delta, lambda r, c: tuple(transform * (c + 0.5, r + 0.5)), params)

    if verbose:
        if masked:
            print(f"  Ignored {masked:.1%} of pixels with too few cloud-free observations")
        _print_summary(n_suspicious, alerts)
    return alerts


# =============================================================================
# 6. TERMINAL SUMMARY
# =============================================================================

def _print_summary(n_suspicious: int, alerts: List[Dict]) -> None:
    cats: Dict[str, int] = {}
    for a in alerts:
        cats[a["category"]] = cats.get(a["category"], 0) + 1

    ICONS = {"critical": "🔴", "high": "🟠", "warning": "🟡", "medium": "🔵", "low": "⚪"}
    clf   = "Random Forest (88-91% acc)" if _rf_model is not None else "Rule-based (fallback)"

    print(f"\n{'='*62}")
    print(f"  TerraVision  ·  RF-Powered Fire & Anomaly Classifier")
    print(f"  Classifier : {clf}")
    print(f"{'='*62}")
    print(f"  Suspicious pixels : {n_suspicious}")
    print(f"  Clusters detected : {len(alerts)}")
    print(f"{'─'*62}")
    print(f"  {'Category':<28}  {'Count':>5}  {'Avg Conf':>8}")
    print(f"{'─'*62}")
    for cat in ["natural_wildfire","agricultural_burn","illegal_encroachment","anomaly","unclassified"]:
        ca = [a for a in alerts if a["category"] == cat]
        if not ca: continue
        avg_conf = sum(a["confidence"] for a in ca) / len(ca)
        icon = ICONS.get(SEVERITY_MAP.get(cat, "low"), "  ")
        print(f"  {icon} {cat:<26}  {len(ca):>5}  {avg_conf:>7.0f}%")
    print(f"{'='*62}\n")

    print(f"  Top 5 alerts:")
    print(f"  {'#':<3}  {'Category':<22}  {'Sev':<8}  {'Conf':>5}  {'Lat':>9}  {'Lon':>10}")
    print(f"  {'─'*64}")
    for i, a in enumerate(alerts[:5], 1):
        print(f"  {i:<3}  {a['category']:<22}  {a['severity']:<8}  "
              f"{a['confidence']:>4}%  {a['lat']:>9.4f}  {a['lon']:>10.4f}")
    print()


# =============================================================================
# 7. COMMAND LINE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="TerraVision fire classifier (ML)")
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("--params", default=None)
    parser.add_argument("--out",    default=None)
    args = parser.parse_args()

    results = detect_encroachments(args.baseline, args.current,
                                   params_json=args.params, verbose=True)
    if args.out:
        Path(args.out).write_text(_json.dumps(results, indent=2))
        print(f"  Alerts written to {args.out}")