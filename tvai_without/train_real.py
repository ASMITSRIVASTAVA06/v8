"""
TerraVision AI — train the classifier on real, independently labelled data.

  python train_real.py                  # uses training/real_dataset.csv (+ your reviewed alerts)
  python train_real.py --force          # save even if the quality gate fails
  python train_real.py --db terravision.db

What it does, and why it can be trusted more than the synthetic model:
  * Spatial cross-validation: clusters from the same ~1-degree cell never sit in both the training
    and the test fold, so the score reflects performance on places the model has not seen.
  * Compares a Random Forest with gradient boosting and keeps the better one.
  * Calibrates probabilities (sigmoid, grouped folds) so "80 %" means about 80 %.
  * Scores the old synthetic model on the same real labels, so you can see the improvement.
  * Quality gate: the new model replaces the old one only if it clears a minimum score and beats it.
  * Merges clusters you verified in the Alerts section (weight 2) with the automatic labels.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import modelcard
from model import CLASS_NAMES, FEATURE_NAMES, MODEL_PATH
from training_data import DATASET_CSV

MIN_SAMPLES = 300
MIN_PER_CLASS = 30
MIN_GROUPS = 6
GATE_MACRO_F1 = 0.55
ANALYST_WEIGHT = 2.0
SYNTHETIC_BACKUP = Path("terravision_rf_synthetic.joblib")


# ── data ─────────────────────────────────────────────────────────────────────

def load_dataset(path: Path = DATASET_CSV) -> Dict:
    """X, y (0..3), sample weights and spatial groups from the CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label") in CLASS_NAMES]
    return {
        "X": np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=np.float32),
        "y": np.array([CLASS_NAMES.index(r["label"]) for r in rows], dtype=np.int32),
        "w": np.array([float(r.get("weight") or 1.0) for r in rows]),
        "groups": np.array([spatial_group(float(r["lat"]), float(r["lon"]), r.get("region", "")) for r in rows]),
    }


def spatial_group(lat: float, lon: float, region: str = "") -> str:
    """Region plus 1-degree cell: the unit that must not be split between training and test."""
    return f"{region}|{int(np.floor(lat))}|{int(np.floor(lon))}"


def analyst_rows(alerts: List[Dict]) -> Dict:
    """Reviewed alerts as extra training rows: a verified alert keeps its category, true_category overrides."""
    X, y, w, g = [], [], [], []
    for a in alerts:
        label = a.get("true_category") or (a.get("category") if a.get("status") == "verified" else None)
        shape = a.get("shape") or {}
        if label not in CLASS_NAMES or not shape:
            continue
        feats = {"delta_mean": a.get("delta_mean"), "delta_max": a.get("delta_max"), "delta_std": a.get("delta_std"),
                 "cluster_size": a.get("cluster_size"), "bbox_area": a.get("bbox_area"), **shape}
        if any(feats.get(k) is None for k in FEATURE_NAMES):
            continue
        X.append([float(feats[k]) for k in FEATURE_NAMES]); y.append(CLASS_NAMES.index(label))
        w.append(ANALYST_WEIGHT); g.append(spatial_group(a["lat"], a["lon"], a.get("region", "")))
    return {"X": np.array(X, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)), "y": np.array(y, dtype=np.int32),
            "w": np.array(w), "groups": np.array(g)}


def merge(*parts: Dict) -> Dict:
    parts = [p for p in parts if len(p["y"])]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "w", "groups")}


def check_enough(d: Dict) -> Optional[str]:
    """None when the data can support training, otherwise a plain-language reason."""
    n = len(d["y"])
    if n < MIN_SAMPLES:
        return f"only {n} labelled clusters (need {MIN_SAMPLES}); run training_data.py with more regions or windows"
    counts = {CLASS_NAMES[c]: int((d["y"] == c).sum()) for c in range(len(CLASS_NAMES))}
    thin = [f"{k} ({v})" for k, v in counts.items() if v < MIN_PER_CLASS]
    if thin:
        return f"too few examples of: {', '.join(thin)} (need {MIN_PER_CLASS} each)"
    if len(set(d["groups"])) < MIN_GROUPS:
        return f"only {len(set(d['groups']))} distinct places (need {MIN_GROUPS}) for spatial validation"
    return None


# ── models and validation ────────────────────────────────────────────────────

def candidates() -> Dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    return {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, class_weight="balanced", random_state=42),
    }


def spatial_folds(groups: np.ndarray, y: np.ndarray, k: int = 5):
    from sklearn.model_selection import GroupKFold
    k = max(2, min(k, len(set(groups))))
    return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))


def cv_predict(make, d: Dict, folds) -> np.ndarray:
    """Out-of-fold class probabilities (n, 4); every row is predicted by a model that never saw its place."""
    proba = np.zeros((len(d["y"]), len(CLASS_NAMES)))
    for tr, te in folds:
        m = make()
        m.fit(d["X"][tr], d["y"][tr], sample_weight=d["w"][tr])
        proba[np.ix_(te, [int(c) for c in m.classes_])] = m.predict_proba(d["X"][te])
    return proba


def scores(y: np.ndarray, proba: np.ndarray) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score
    pred = proba.argmax(axis=1)
    onehot = np.eye(len(CLASS_NAMES))[y]
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(CLASS_NAMES)), zero_division=0)),
            "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
            "ece": expected_calibration_error(y, proba)}


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def per_class_report(y: np.ndarray, proba: np.ndarray) -> Dict:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, n = precision_recall_fscore_support(y, proba.argmax(axis=1), labels=range(len(CLASS_NAMES)), zero_division=0)
    return {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": int(n[i])}
            for i in range(len(CLASS_NAMES))}


def score_existing_model(path: Path, d: Dict) -> Optional[Dict]:
    """How the model currently on disk (usually the synthetic one) does on the real labels."""
    try:
        import joblib
        m = joblib.load(path)
        proba = np.zeros((len(d["y"]), len(CLASS_NAMES)))
        proba[:, [int(c) for c in m.classes_]] = m.predict_proba(d["X"])
        return scores(d["y"], proba)
    except Exception:
        return None


def fit_calibrated(make, d: Dict, folds):
    """Final model fitted on everything, with probabilities calibrated on grouped folds."""
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(make(), method="sigmoid", cv=folds)
    cal.fit(d["X"], d["y"], sample_weight=d["w"])
    return cal


# ── driver ───────────────────────────────────────────────────────────────────

def train_from_data(d: Dict, force: bool = False, model_path: Path = MODEL_PATH,
                    log=print, existing_score: Optional[Dict] = None) -> Tuple[Optional[object], Dict]:
    """Train, validate, gate, save. Returns (model or None if rejected, report dict)."""
    problem = check_enough(d)
    if problem:
        return None, {"accepted": False, "reason": problem}

    folds = spatial_folds(d["groups"], d["y"])
    results, oof = {}, {}
    for name, make in candidates().items():
        oof[name] = cv_predict(make, d, folds)
        results[name] = scores(d["y"], oof[name])
        log(f"  {name:<18} spatial-CV macro-F1 {results[name]['macro_f1']:.3f}  accuracy {results[name]['accuracy']:.3f}")
    best = max(results, key=lambda k: results[k]["macro_f1"])
    metrics = results[best]
    report = {"accepted": False, "best": best, "metrics": metrics, "candidates": results,
              "per_class": per_class_report(d["y"], oof[best]), "baseline": existing_score}

    if not force:
        if metrics["macro_f1"] < GATE_MACRO_F1:
            report["reason"] = f"macro-F1 {metrics['macro_f1']:.2f} is below the {GATE_MACRO_F1:.2f} quality gate"
            return None, report
        if existing_score and metrics["macro_f1"] <= existing_score["macro_f1"]:
            report["reason"] = (f"no better than the current model on real labels "
                                f"({metrics['macro_f1']:.2f} vs {existing_score['macro_f1']:.2f})")
            return None, report

    model = fit_calibrated(candidates()[best], d, folds)
    import joblib
    if Path(model_path).exists() and not SYNTHETIC_BACKUP.exists():
        shutil.copy(model_path, SYNTHETIC_BACKUP)           # keep the synthetic model once
    joblib.dump(model, model_path)
    modelcard.write({"provenance": "real", "algorithm": best, "n_samples": int(len(d["y"])),
                     "class_counts": {CLASS_NAMES[c]: int((d["y"] == c).sum()) for c in range(len(CLASS_NAMES))},
                     "n_places": int(len(set(d["groups"]))), "validation": "spatial (grouped by region + 1-degree cell)",
                     "metrics": metrics, "per_class": report["per_class"],
                     "labels": "independent: FIRMS fire, ESA WorldCover, Hansen forest loss, Sentinel-2 NDVI, WDPA"})
    report["accepted"] = True
    return model, report


def format_report(r: Dict) -> str:
    L = []
    if not r.get("accepted"):
        L.append(f"NOT SAVED: {r.get('reason', 'rejected')}")
    if "metrics" in r:
        m = r["metrics"]
        L.append(f"Chosen: {r['best']}  macro-F1 {m['macro_f1']:.3f}  accuracy {m['accuracy']:.3f}  "
                 f"Brier {m['brier']:.3f}  ECE {m['ece']:.3f}   (spatial cross-validation)")
        if r.get("baseline"):
            b = r["baseline"]
            L.append(f"Current model on the same real labels: macro-F1 {b['macro_f1']:.3f}  accuracy {b['accuracy']:.3f}")
        L.append("Per class:")
        for k, v in r["per_class"].items():
            L.append(f"  {k:<22} precision {v['precision']:.2f}  recall {v['recall']:.2f}  n={v['n']}")
    if r.get("accepted"):
        L.append(f"Saved {MODEL_PATH} and model_card.json. Restart the app to use it.")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train the TerraVision classifier on real labelled data.")
    p.add_argument("--csv", default=str(DATASET_CSV))
    p.add_argument("--db", help="terravision.db with reviewed alerts to merge in")
    p.add_argument("--force", action="store_true", help="save even if the quality gate fails")
    args = p.parse_args(argv)

    if not Path(args.csv).exists():
        print(f"No dataset at {args.csv}. Build one first:  python training_data.py --regions all --windows 6")
        return 1
    d = load_dataset(Path(args.csv))
    try:
        import store
        alerts = [a for s in store.latest_scans(Path(args.db) if args.db else None) for a in s["alerts"]]
        extra = analyst_rows(alerts)
        if len(extra["y"]):
            print(f"Merging {len(extra['y'])} analyst-reviewed alerts (weight {ANALYST_WEIGHT}).")
            d = merge(d, extra)
    except Exception as e:
        print(f"(no reviewed alerts merged: {type(e).__name__})")

    baseline = score_existing_model(MODEL_PATH, d) if MODEL_PATH.exists() else None
    model, report = train_from_data(d, force=args.force, existing_score=baseline)
    print(format_report(report))
    return 0 if model is not None else 2


if __name__ == "__main__":
    sys.exit(main())
