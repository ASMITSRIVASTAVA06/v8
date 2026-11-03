"""
TerraVision AI — alert helpers (no Streamlit dependency, so they are easy to test).

Everything the Dashboard and Alerts sections share lives here:
  * category / severity styling constants
  * turning raw detector output into an enriched DataFrame (stable IDs, region, timestamp)
  * exporting alerts as CSV, JSON, a printable HTML report or a plain-text summary
  * sharing alerts by e-mail (SMTP with attachments, mailto: and Gmail compose links)
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd

from evidence import zone_of

try:  # .env is already used by gee_gateway.py; reuse it for SMTP settings
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv missing is not fatal
    pass

# =============================================================================
# Styling constants (shared by app.py, dashboard.py and alerts_view.py)
# =============================================================================

CATEGORY_STYLE = {
    "illegal_encroachment": {"color": "#ff4d6d", "label": "Illegal encroachment"},
    "natural_wildfire":     {"color": "#ff8a3d", "label": "Natural wildfire"},
    "agricultural_burn":    {"color": "#facc15", "label": "Agricultural burn"},
    "anomaly":              {"color": "#a5b4d4", "label": "Unclassified anomaly"},
    "unclassified":         {"color": "#64748b", "label": "Unclassified"},
}

SEVERITY_ORDER = ["critical", "high", "warning", "medium", "low"]
SEVERITY_COLOR = {
    "critical": "#ef4444",
    "high":     "#f97316",
    "warning":  "#eab308",
    "medium":   "#38bdf8",
    "low":      "#64748b",
}

LIFECYCLES = ["new", "recurring"]
VERDICT_ORDER = ["Strong evidence", "Possible", "Likely false positive"]
VERDICT_COLOR = {"Strong evidence": "#34d399", "Possible": "#fbbf24", "Likely false positive": "#8aa0c4"}

# Features the Random Forest in model.py is trained on (same order as model.FEATURE_NAMES)
MODEL_FEATURES = [
    "delta_mean", "delta_max", "delta_std", "cluster_size", "aspect_ratio", "compactness",
    "convexity", "edge_ratio", "spread_rate", "interior_uniformity", "bbox_area",
]
MODEL_CLASSES = ["natural_wildfire", "agricultural_burn", "illegal_encroachment", "anomaly"]

SHAPE_FEATURES = [
    "aspect_ratio", "compactness", "convexity",
    "edge_ratio", "spread_rate", "interior_uniformity",
]

PIXEL_SIZE_M = 500                             # VIIRS pixel size used by the scan

EXPORT_COLUMNS = [
    "alert_id", "region", "detected_at", "category", "severity", "confidence", "model_confidence",
    "verdict", "lifecycle", "times_seen", "status", "assignee", "note",
    "in_protected_area", "protected_area", "builtup_fraction", "ndvi_change", "tree_loss_fraction",
    "fire_detected", "evidence_text", "lat", "lon", "cluster_size", "delta_mean", "delta_max", "maps_url",
]


def category_label(cat: str) -> str:
    return CATEGORY_STYLE.get(cat, {}).get("label", str(cat).replace("_", " ").capitalize())


def category_color(cat: str) -> str:
    return CATEGORY_STYLE.get(cat, CATEGORY_STYLE["unclassified"])["color"]


# =============================================================================
# Loading / enriching alerts
# =============================================================================

def alert_id(region: str, lat: float, lon: float, category: str) -> str:
    """Short, stable ID so the same cluster keeps its ID across re-scans."""
    raw = f"{region}|{lat:.4f}|{lon:.4f}|{category}".encode()
    return "TV-" + hashlib.sha1(raw).hexdigest()[:8].upper()


def maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}"


def pixel_footprint(lat: float, lon: float, size_m: float = PIXEL_SIZE_M) -> List[List[float]]:
    """[[south, west], [north, east]] of one satellite pixel centred on the alert."""
    import math
    half = size_m / 2
    dlat = half / 111_320
    dlon = half / (111_320 * max(math.cos(math.radians(lat)), 0.01))
    return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]


def format_dms(value: float, positive: str, negative: str) -> str:
    hemi = positive if value >= 0 else negative
    v = abs(value)
    d = int(v)
    m = int((v - d) * 60)
    sec = (v - d - m / 60) * 3600
    return f"{d}°{m:02d}'{sec:04.1f}\"{hemi}"


def to_dataframe(alerts: Iterable[Dict]) -> pd.DataFrame:
    """Flatten alerts (including nested shape / scores) into one tidy DataFrame."""
    rows = []
    for a in alerts:
        row = {
            "alert_id": a.get("alert_id") or alert_id(a.get("region", ""), a["lat"], a["lon"], a["category"]),
            "region": a.get("region", ""),
            "detected_at": a.get("detected_at", ""),
            "category": a["category"],
            "category_label": category_label(a["category"]),
            "severity": a["severity"],
            "confidence": a["confidence"],
            "lat": a["lat"],
            "lon": a["lon"],
            "cluster_size": a["cluster_size"],
            "delta_mean": a["delta_mean"],
            "delta_max": a["delta_max"],
            "maps_url": maps_url(a["lat"], a["lon"]),
            "model_confidence": a.get("model_confidence", a["confidence"]),
            "verdict": a.get("verdict", ""),
            "lifecycle": a.get("lifecycle", ""),
            "times_seen": a.get("times_seen", 1),
            "first_seen": a.get("first_seen", ""),
            "period": a.get("period", ""),
            "in_protected_area": a.get("in_protected_area"),
            "in_official_boundary": a.get("in_official_boundary"),
            "zone": a.get("zone") or zone_of(a),
            "tree_context": a.get("tree_context"),
            "protected_area": a.get("protected_area") or "",
            "builtup_fraction": a.get("builtup_fraction"),
            "tree_fraction": a.get("tree_fraction"),
            "ndvi_change": a.get("ndvi_change"),
            "tree_loss_fraction": a.get("tree_loss_fraction"),
            "fire_detected": a.get("fire_detected"),
            "corroboration": a.get("corroboration", 0),
            "urban_light": bool(a.get("urban_light", False)),
            "evidence_text": "; ".join(a.get("evidence", [])),
            "status": a.get("status", "new"),
            "assignee": a.get("assignee", ""),
            "note": a.get("note", ""),
            "true_category": a.get("true_category", ""),
        }
        for k in SHAPE_FEATURES:
            row[f"shape_{k}"] = (a.get("shape") or {}).get(k)
        for k, v in (a.get("scores") or {}).items():
            row[f"score_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        rank = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        df["_rank"] = df["severity"].map(rank).fillna(len(rank))
        df = df.sort_values(["_rank", "confidence"], ascending=[True, False]).drop(columns="_rank")
        df = df.reset_index(drop=True)
    return df


# =============================================================================
# Exports
# =============================================================================

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    return df[cols].to_csv(index=False).encode("utf-8")


def to_json_bytes(df: pd.DataFrame, alerts: Iterable[Dict]) -> bytes:
    """Export the full original alert records (with shape + scores) for the given rows."""
    wanted = set(df["alert_id"]) if not df.empty else set()
    records = [a for a in alerts if a.get("alert_id") in wanted]
    order = {aid: i for i, aid in enumerate(df["alert_id"])} if not df.empty else {}
    records.sort(key=lambda a: order.get(a.get("alert_id"), 0))
    return json.dumps(records, indent=2).encode("utf-8")


def counts_by(df: pd.DataFrame, column: str, order: Optional[List[str]] = None) -> Dict[str, int]:
    if df.empty:
        return {}
    counts = df[column].value_counts().to_dict()
    if order:
        return {k: int(counts[k]) for k in order if k in counts}
    return {k: int(v) for k, v in counts.items()}


def build_summary_text(df: pd.DataFrame, title: str = "TerraVision AI alert summary", top_n: int = 10) -> str:
    """Plain-text summary used for e-mail bodies and copy/paste sharing."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [title, f"Generated {now}", ""]
    if df.empty:
        lines.append("No alerts match the current selection.")
        return "\n".join(lines)

    lines.append(f"Total alerts: {len(df)}")
    sev = counts_by(df, "severity", SEVERITY_ORDER)
    lines.append("By severity: " + ", ".join(f"{k} {v}" for k, v in sev.items()))
    cat = counts_by(df, "category")
    lines.append("By type: " + ", ".join(f"{category_label(k)} {v}" for k, v in cat.items()))
    regions = sorted(df["region"].unique())
    if regions:
        lines.append("Region: " + ", ".join(regions))
    lines += ["", f"Top {min(top_n, len(df))} alerts (most severe first):"]
    for _, r in df.head(top_n).iterrows():
        lines.append(
            f"- {r['alert_id']} | {r['severity'].upper()} | {category_label(r['category'])} | "
            f"{r['confidence']}% confidence{(' | ' + r['verdict']) if r['verdict'] else ''}"
            f"{(' | ' + r['lifecycle']) if r['lifecycle'] else ''} | {r['cluster_size']} px | "
            f"{r['lat']:.4f}, {r['lon']:.4f} | {r['maps_url']}"
        )
    if len(df) > top_n:
        lines.append(f"... and {len(df) - top_n} more (see attachment).")
    lines += [
        "",
        "Prototype output: verify every alert with an on-the-ground check before acting.",
    ]
    return "\n".join(lines)


def build_html_report(df: pd.DataFrame, title: str = "TerraVision AI — Alert Report", max_rows: int = 500) -> bytes:
    """Self-contained HTML report. Open in a browser; Print → Save as PDF for a PDF copy."""
    esc = html.escape
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def badge(sev: str) -> str:
        c = SEVERITY_COLOR.get(sev, "#78909c")
        fg = "#222" if sev == "warning" else "#fff"
        return f'<span class="b" style="background:{c};color:{fg}">{esc(sev.upper())}</span>'

    sev_counts = counts_by(df, "severity", SEVERITY_ORDER)
    cat_counts = counts_by(df, "category")

    cards = "".join(
        f'<div class="card"><div class="n" style="color:{SEVERITY_COLOR.get(k, "#555")}">{v}</div>'
        f'<div class="l">{esc(k)}</div></div>'
        for k, v in sev_counts.items()
    )
    cat_list = "".join(
        f'<li><span class="dot" style="background:{category_color(k)}"></span>{esc(category_label(k))}: <b>{v}</b></li>'
        for k, v in cat_counts.items()
    )

    shown = df.head(max_rows)
    body_rows = "".join(
        "<tr>"
        f"<td>{esc(str(r.alert_id))}</td>"
        f"<td>{esc(str(r.region))}</td>"
        f"<td>{esc(category_label(r.category))}</td>"
        f"<td>{badge(r.severity)}</td>"
        f"<td>{r.confidence}%</td>"
        f"<td>{r.cluster_size}</td>"
        f"<td>{r.delta_mean}</td><td>{r.delta_max}</td>"
        f'<td><a href="{esc(r.maps_url)}">{r.lat:.4f}, {r.lon:.4f}</a></td>'
        "</tr>"
        for r in shown.itertuples()
    )
    truncated = (
        f"<p class='note'>Showing the first {max_rows} of {len(df)} alerts. Use the CSV/JSON export for the full list.</p>"
        if len(df) > max_rows else ""
    )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
 body{{font-family:Segoe UI,Helvetica,Arial,sans-serif;color:#1b2b22;margin:32px;}}
 h1{{color:#1d63ed;margin:0 0 4px}} .sub{{color:#5b6f64;margin-bottom:20px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
 .card{{border:1px solid #cfe3d7;border-radius:10px;padding:12px 18px;min-width:90px}}
 .n{{font-size:1.7rem;font-weight:600}} .l{{font-size:.75rem;text-transform:uppercase;color:#5b6f64;letter-spacing:.6px}}
 ul{{list-style:none;padding:0}} li{{margin:4px 0}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}}
 table{{border-collapse:collapse;width:100%;font-size:.82rem;margin-top:12px}}
 th,td{{border-bottom:1px solid #e1ece5;padding:6px 8px;text-align:left}}
 th{{background:#eef6f1;position:sticky;top:0}}
 .b{{padding:2px 8px;border-radius:10px;font-size:.7rem;font-weight:600}}
 .note{{color:#5b6f64;font-size:.8rem}}
 @media print{{body{{margin:12mm}} th{{position:static}}}}
</style></head><body>
<h1>{esc(title)}</h1>
<div class="sub">Generated {esc(now)} · {len(df)} alert(s)</div>
<div class="cards">{cards}</div>
<ul>{cat_list}</ul>
<table><thead><tr><th>ID</th><th>Region</th><th>Type</th><th>Severity</th><th>Conf.</th>
<th>Cluster (px)</th><th>Δ mean</th><th>Δ peak</th><th>Location</th></tr></thead>
<tbody>{body_rows}</tbody></table>
{truncated}
<p class="note">Radiance values in nW/cm²/sr. Prototype output — verify every alert with an on-the-ground check before acting.</p>
</body></html>"""
    return doc.encode("utf-8")


# =============================================================================
# Sharing by e-mail
# =============================================================================

_EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")


def parse_recipients(text: str) -> tuple[List[str], List[str]]:
    """Split a comma / semicolon / newline separated string into (valid, invalid)."""
    parts = [p.strip() for p in re.split(r"[,;\n]", text or "") if p.strip()]
    valid = [p for p in parts if _EMAIL_RE.match(p)]
    invalid = [p for p in parts if not _EMAIL_RE.match(p)]
    return valid, invalid


def get_smtp_config(secrets=None) -> Optional[Dict]:
    """
    SMTP settings from environment variables (.env) or Streamlit secrets.
    Needs SMTP_HOST, SMTP_USER and SMTP_PASSWORD; SMTP_PORT (587), SMTP_FROM
    and SMTP_SECURITY (starttls | ssl) are optional.
    """
    def get(key, default=None):
        val = os.environ.get(key)
        if val:
            return val
        if secrets is not None:
            try:
                return secrets.get(key, default)
            except Exception:
                return default
        return default

    host, user, pwd = get("SMTP_HOST"), get("SMTP_USER"), get("SMTP_PASSWORD")
    if not (host and user and pwd):
        return None
    port = int(get("SMTP_PORT", 587))
    security = str(get("SMTP_SECURITY", "ssl" if port == 465 else "starttls")).lower()
    return {
        "host": host, "port": port, "user": user, "password": pwd,
        "sender": get("SMTP_FROM", user), "security": security,
    }


def send_alert_email(
    config: Dict,
    recipients: List[str],
    subject: str,
    body: str,
    attachments: Dict[str, tuple[bytes, str]],
) -> None:
    """
    Send an e-mail with attachments. `attachments` maps filename -> (bytes, mime "type/subtype").
    Raises on failure so the caller can show the real error.
    """
    msg = EmailMessage()
    msg["From"] = config["sender"]
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)
    for name, (data, mime) in attachments.items():
        maintype, _, subtype = mime.partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)

    context = ssl.create_default_context()
    if config["security"] == "ssl":
        with smtplib.SMTP_SSL(config["host"], config["port"], context=context, timeout=30) as s:
            s.login(config["user"], config["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(config["host"], config["port"], timeout=30) as s:
            s.starttls(context=context)
            s.login(config["user"], config["password"])
            s.send_message(msg)


def mailto_link(recipients: List[str], subject: str, body: str, max_body: int = 1500) -> str:
    """mailto: link (opens the user's mail app). Attachments are not possible with mailto."""
    if len(body) > max_body:
        body = body[:max_body].rsplit("\n", 1)[0] + "\n... (truncated — download the full report and attach it)"
    return f"mailto:{','.join(recipients)}?subject={quote(subject)}&body={quote(body)}"


def gmail_link(recipients: List[str], subject: str, body: str, max_body: int = 1500) -> str:
    if len(body) > max_body:
        body = body[:max_body].rsplit("\n", 1)[0] + "\n... (truncated — download the full report and attach it)"
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote(','.join(recipients))}&su={quote(subject)}&body={quote(body)}"
    )


# =============================================================================
# Training labels for model.py  (python model.py --csv terravision_labels.csv)
# =============================================================================

def build_training_csv(alerts: Iterable[Dict]) -> Tuple[bytes, int, int]:
    """
    Turn analyst decisions into a labelled CSV the model can retrain on.

    * status "verified"        -> label = the confirmed type, else the model's own category
    * status "false positive"  -> only used if the analyst also set the true type
    Everything else (and rows from older scans lacking delta_std / bbox_area) is skipped.
    Returns (csv_bytes, rows_written, rows_skipped_among_labelled).
    """
    rows, skipped = [], 0
    for a in alerts:
        st = a.get("status")
        if st not in ("verified", "false positive"):
            continue
        label = a.get("true_category") or (a.get("category") if st == "verified" else "")
        shape = a.get("shape") or {}
        feats = {**{k: a.get(k) for k in ("delta_mean", "delta_max", "delta_std", "cluster_size", "bbox_area")},
                 **{k: shape.get(k) for k in SHAPE_FEATURES}}
        if label not in MODEL_CLASSES or any(feats.get(f) is None for f in MODEL_FEATURES):
            skipped += 1
            continue
        rows.append({**{f: feats[f] for f in MODEL_FEATURES}, "label": label})
    df = pd.DataFrame(rows, columns=MODEL_FEATURES + ["label"])
    return df.to_csv(index=False).encode("utf-8"), len(rows), skipped


# =============================================================================
# Telegram push (for field staff)
# =============================================================================

def get_telegram_config(secrets=None) -> Optional[Dict]:
    def get(key):
        val = os.environ.get(key)
        if val:
            return val
        if secrets is not None:
            try:
                return secrets.get(key)
            except Exception:
                return None
        return None
    token, chat = get("TELEGRAM_BOT_TOKEN"), get("TELEGRAM_CHAT_ID")
    return {"token": token, "chat_id": chat} if token and chat else None


def send_telegram(config: Dict, text: str) -> None:
    """Send a text message through the Telegram Bot API. Raises on failure."""
    import urllib.parse
    import urllib.request
    data = urllib.parse.urlencode({
        "chat_id": config["chat_id"], "text": text[:4000], "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{config['token']}/sendMessage", data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Telegram returned HTTP {resp.status}")
