"""
TerraVision AI — calibration report.

Answers the question "when TerraVision says 80 %, is it right about 80 % of the time?"
using the alerts you have reviewed in the Alerts section (status verified / false positive).

  python calibrate.py                         # reads terravision.db
  python calibrate.py --db path/to/terravision.db
  python calibrate.py --json terravision_alerts.json   # alerts downloaded from the Alerts section

It reports precision overall and by evidence verdict, a reliability table by confidence band,
the Brier score and expected calibration error (ECE) of the model's own confidence versus the
evidence-adjusted confidence, and — once there are enough labels — a suggested re-weighting of
the evidence signals with a cross-validated check. Nothing is changed automatically.

Honesty note: a calibration is only as good as the sample. Review alerts across ALL confidence
levels, not just the top ones, or the numbers will look better than reality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BINS = [0, 50, 60, 70, 80, 90, 101]
BIN_LABELS = ["<50", "50-59", "60-69", "70-79", "80-89", "90+"]
MIN_FIT_LABELS = 60
MIN_PER_CLASS = 15


# ── labels ───────────────────────────────────────────────────────────────────

def is_correct(alert: Dict) -> Optional[int]:
    """1 = analyst confirmed the alert as classified, 0 = wrong, None = not reviewed / undecidable."""
    status = alert.get("status")
    true_cat = alert.get("true_category") or ""
    cat = alert.get("category")
    if true_cat and true_cat != cat:
        return 0                                   # analyst says it is something else
    if status == "verified":
        return 1
    if status == "false positive":
        return 0
    return None


def labelled_frame(alerts: List[Dict]) -> pd.DataFrame:
    rows = []
    for a in alerts:
        y = is_correct(a)
        if y is None:
            continue
        rows.append({
            "alert_id": a.get("alert_id"), "category": a.get("category"), "verdict": a.get("verdict") or "none",
            "y": y,
            "confidence": a.get("confidence"),
            "model_confidence": a.get("model_confidence", a.get("confidence")),
            "builtup_fraction": a.get("builtup_fraction"), "urban": float(bool(a.get("urban_light"))),
            "in_pa": a.get("in_protected_area"), "ndvi_change": a.get("ndvi_change"),
            "tree_loss": a.get("tree_loss_fraction"), "fire": a.get("fire_detected"),
            "times_seen": a.get("times_seen", 1),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["confidence", "model_confidence"]).drop_duplicates("alert_id", keep="last")


# ── metrics ──────────────────────────────────────────────────────────────────

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability_table(conf: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Per confidence band: how many alerts, what the model claimed, what actually happened."""
    band = pd.cut(conf, BINS, labels=BIN_LABELS, right=False)
    df = pd.DataFrame({"band": band, "conf": conf / 100.0, "y": y})
    t = df.groupby("band", observed=True).agg(n=("y", "size"), claimed=("conf", "mean"), actual=("y", "mean"))
    t["gap"] = t["actual"] - t["claimed"]            # negative = over-confident
    return t.reset_index()


def ece(conf: pd.Series, y: pd.Series) -> float:
    """Expected calibration error: the average gap between claimed and actual accuracy, weighted by band size."""
    t = reliability_table(conf, y)
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum()) if len(t) else float("nan")


def by_verdict(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("verdict").agg(n=("y", "size"), precision=("y", "mean"))
              .reset_index().sort_values("precision", ascending=False))


def sample_warnings(df: pd.DataFrame) -> List[str]:
    out = []
    n = len(df)
    if n < 30:
        out.append(f"Only {n} labelled alerts: treat every number below as a rough indication.")
    if n:
        high = float((df["confidence"] >= 80).mean())
        if high > 0.7:
            out.append(f"{high:.0%} of the reviewed alerts have confidence ≥ 80 %: the sample is biased towards "
                       "the best alerts, so precision will look better than it is. Review some lower-ranked ones too.")
        if df["y"].nunique() < 2:
            out.append("All reviewed alerts have the same outcome, so calibration cannot be measured yet.")
    return out


FEATURES = ["model_conf", "urban", "builtup", "in_pa", "in_pa_known", "ndvi_drop", "tree_loss", "fire", "persistent"]


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["model_conf"] = df["model_confidence"] / 100.0
    x["urban"] = df["urban"]
    x["builtup"] = df["builtup_fraction"].astype(float).fillna(0.0)
    x["in_pa"] = df["in_pa"].map({True: 1.0, False: 0.0}).fillna(0.0)
    x["in_pa_known"] = df["in_pa"].notna().astype(float)
    x["ndvi_drop"] = (-df["ndvi_change"].astype(float)).clip(lower=0).fillna(0.0)
    x["tree_loss"] = df["tree_loss"].astype(float).fillna(0.0)
    x["fire"] = df["fire"].map({True: 1.0, False: 0.0}).fillna(0.0)
    x["persistent"] = (df["times_seen"].astype(float) >= 2).astype(float)
    return x[FEATURES]


def suggest_weights(df: pd.DataFrame) -> Dict:
    """Fit a logistic regression on the evidence signals and check it by cross-validation."""
    n_pos, n_neg = int(df["y"].sum()), int((1 - df["y"]).sum())
    if len(df) < MIN_FIT_LABELS or min(n_pos, n_neg) < MIN_PER_CLASS:
        return {"ok": False, "reason": f"need at least {MIN_FIT_LABELS} labels with ≥ {MIN_PER_CLASS} of each outcome "
                                       f"(have {len(df)}: {n_pos} correct, {n_neg} wrong)"}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x, y = feature_matrix(df), df["y"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    p_full = cross_val_predict(lr, x, y, cv=cv, method="predict_proba")[:, 1]
    p_base = cross_val_predict(lr, x[["model_conf"]], y, cv=cv, method="predict_proba")[:, 1]
    lr.fit(x, y)
    return {
        "ok": True,
        "coefficients": dict(zip(FEATURES, (float(c) for c in lr.coef_[0]))),
        "auc_model_only": float(roc_auc_score(y, p_base)),
        "auc_with_evidence": float(roc_auc_score(y, p_full)),
        "brier_with_evidence": brier(p_full, y.to_numpy()),
        "brier_model_only": brier(p_base, y.to_numpy()),
    }


def analyse(alerts: List[Dict]) -> Dict:
    df = labelled_frame(alerts)
    result: Dict = {"n": len(df), "warnings": sample_warnings(df), "frame": df}
    if df.empty:
        return result
    y = df["y"].to_numpy(dtype=float)
    result.update(
        precision=float(df["y"].mean()),
        by_verdict=by_verdict(df),
        table_adjusted=reliability_table(df["confidence"], df["y"]),
        table_model=reliability_table(df["model_confidence"], df["y"]),
        brier_adjusted=brier(df["confidence"].to_numpy() / 100, y),
        brier_model=brier(df["model_confidence"].to_numpy() / 100, y),
        ece_adjusted=ece(df["confidence"], df["y"]),
        ece_model=ece(df["model_confidence"], df["y"]),
        fit=suggest_weights(df),
    )
    return result


# ── report ───────────────────────────────────────────────────────────────────

def format_report(r: Dict) -> str:
    L = ["TerraVision calibration report", "=" * 30, f"Reviewed alerts with a usable outcome: {r['n']}"]
    for w in r["warnings"]:
        L.append(f"WARNING: {w}")
    if not r["n"]:
        L.append("\nNothing to analyse yet. In the Alerts section, mark alerts verified / false positive first.")
        return "\n".join(L)

    L.append(f"\nOverall precision (alerts that were right): {r['precision']:.0%}")
    L.append("\nPrecision by evidence verdict:")
    L.append(r["by_verdict"].to_string(index=False, formatters={"precision": "{:.0%}".format}))

    L.append("\nReliability by confidence band (claimed vs. actual):")
    for name, key in (("Evidence-adjusted confidence", "table_adjusted"), ("Model confidence only", "table_model")):
        L.append(f"\n  {name}")
        t = r[key].copy()
        L.append(t.to_string(index=False, formatters={"claimed": "{:.0%}".format, "actual": "{:.0%}".format,
                                                       "gap": "{:+.0%}".format}))

    better = r["brier_adjusted"] < r["brier_model"]
    L.append("\nScores (lower is better):")
    L.append(f"  Brier   model {r['brier_model']:.3f}  ->  adjusted {r['brier_adjusted']:.3f}   "
             f"({'evidence helped' if better else 'evidence did NOT help on this sample'})")
    L.append(f"  ECE     model {r['ece_model']:.3f}  ->  adjusted {r['ece_adjusted']:.3f}")

    fit = r["fit"]
    L.append("\nSuggested re-weighting of the evidence signals:")
    if not fit["ok"]:
        L.append(f"  Not enough data yet: {fit['reason']}.")
    else:
        L.append(f"  Cross-validated AUC: model alone {fit['auc_model_only']:.2f} -> with evidence {fit['auc_with_evidence']:.2f}")
        L.append("  Logistic coefficients (positive = raises the chance the alert is real):")
        for k, v in sorted(fit["coefficients"].items(), key=lambda kv: -abs(kv[1])):
            L.append(f"    {k:<12} {v:+.2f}")
        L.append("  Compare with the hand-set rules in evidence.py before changing anything.")
    return "\n".join(L)


# ── input ────────────────────────────────────────────────────────────────────

def load_alerts(db: Optional[str], json_path: Optional[str]) -> List[Dict]:
    if json_path:
        data = json.loads(Path(json_path).read_text())
        return data if isinstance(data, list) else data.get("alerts", [])
    import store
    path = Path(db) if db else None
    return [a for s in store.latest_scans(path) for a in s["alerts"]]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check how trustworthy TerraVision's confidence scores are.")
    p.add_argument("--db", help="path to terravision.db (default: the app's database)")
    p.add_argument("--json", help="alerts JSON downloaded from the Alerts section")
    args = p.parse_args(argv)
    print(format_report(analyse(load_alerts(args.db, args.json))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
