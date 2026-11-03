"""
TerraVision AI — the logic behind the dynamic parts of the interface.

Everything here is plain Python (no Streamlit, no Earth Engine), so it can be tested on its own:
scan-progress stages, sparklines, scan replay, map-click matching, heatmap points, the review
board, the activity feed and the live status strip.
"""

from __future__ import annotations

import html
import math
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ── 1. scan progress ─────────────────────────────────────────────────────────

STAGES = ["Satellite data", "Quality & detection", "Cross-checks", "Boundary & scoring", "Saved"]

_STAGE_KEYS = [
    (0, ("checking for the newest", "downloading", "using cached satellite", "earth engine")),
    (1, ("detecting", "classifying")),
    (2, ("cross-checking",)),
    (3, ("fetching", "boundary", "comparing with the previous", "scoring")),
]


def stage_of(message: str) -> int:
    """Index into STAGES for a progress message from the pipeline (unknown messages keep the last stage)."""
    m = (message or "").lower()
    for idx, keys in _STAGE_KEYS:
        if any(k in m for k in keys):
            return idx
    return -1


class StageTracker:
    """Remembers the furthest stage reached so out-of-order messages never move the bar backwards."""

    def __init__(self) -> None:
        self.index = 0
        self.message = "Starting…"

    def update(self, message: str) -> int:
        self.message = message
        idx = stage_of(message)
        if idx > self.index:
            self.index = idx
        return self.index


def stages_html(index: int, message: str = "", done: bool = False) -> str:
    """A horizontal stepper: finished stages ticked, the current one pulsing."""
    last = len(STAGES) - 1
    if done:
        index = last
    items = []
    for i, name in enumerate(STAGES):
        state = "done" if (i < index or (done and i <= last)) else ("now" if i == index else "todo")
        mark = "✓" if state == "done" else str(i + 1)
        items.append(f'<div class="step {state}"><div class="step-dot">{mark}</div><div class="step-name">{html.escape(name)}</div></div>')
    pct = 100 if done else int(100 * index / last)
    note = f'<div class="step-msg">{html.escape(message)}</div>' if message else ""
    return (f'<div class="stepper"><div class="step-track"><div class="step-fill" style="width:{pct}%"></div></div>'
            f'<div class="steps">{"".join(items)}</div>{note}</div>')


# ── 2. sparklines and scan series ────────────────────────────────────────────

def sparkline_svg(values: Sequence[float], color: str = "#3b82f6", width: int = 110, height: int = 30) -> str:
    """Tiny inline line chart with a dot on the latest value; empty string if there is nothing to draw."""
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad = 3
    pts = [(pad + i * (width - 2 * pad) / (len(vals) - 1), height - pad - (v - lo) / span * (height - 2 * pad))
           for i, v in enumerate(vals)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pts[0][0]:.1f},{height} {path} {pts[-1][0]:.1f},{height}"
    x, y = pts[-1]
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true">'
            f'<polygon points="{area}" fill="{color}" opacity=".16"/>'
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/></svg>')


def scan_series(history, region: Optional[str] = None, category: Optional[str] = None,
                lifecycle: Optional[str] = None, last: int = 8) -> List[int]:
    """Alerts per scan (oldest first) from store.scan_history(); optional region / category / lifecycle filter."""
    if history is None or len(history) == 0:
        return []
    df = history
    if region:
        df = df[df["region"] == region]
    if category:
        df = df[df["category"] == category]
    if lifecycle:
        df = df[df["lifecycle"] == lifecycle]
    if df.empty:
        return []
    per_scan = df.groupby("scan_id")["n"].sum().sort_index()
    return [int(v) for v in per_scan.tolist()[-last:]]


def trend_badge(values: Sequence[int]) -> Tuple[str, str]:
    """(arrow-text, css colour name) comparing the last two values: ('▲ +3', 'bad') etc. Empty if < 2 points."""
    if len(values) < 2:
        return "", ""
    d = values[-1] - values[-2]
    if d == 0:
        return "● no change", "flat"
    return (f"▲ +{d}" if d > 0 else f"▼ {d}"), ("up" if d > 0 else "down")


# ── 3. replay ────────────────────────────────────────────────────────────────

def replay_frames(scans: Iterable[Dict]) -> List[Dict]:
    """Oldest→newest frames {label, scan_id, alerts, n} from saved scans of one region."""
    out = []
    for s in sorted(scans, key=lambda s: s.get("scan_id", 0)):
        out.append({"scan_id": s.get("scan_id"), "label": f'{s.get("period") or "?"} · {s.get("scanned_at", "")[:10]}',
                    "alerts": s.get("alerts", []), "n": len(s.get("alerts", []))})
    return out


def frame_diff(prev: Optional[Dict], cur: Dict, radius_km: float = 1.0) -> Dict[str, int]:
    """How many alerts in `cur` are new, persisted from `prev`, and how many of prev vanished."""
    prev_alerts = prev["alerts"] if prev else []
    persisted = 0
    for a in cur["alerts"]:
        if any(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) <= radius_km for b in prev_alerts):
            persisted += 1
    gone = sum(1 for b in prev_alerts
               if not any(haversine_km(b["lat"], b["lon"], a["lat"], a["lon"]) <= radius_km for a in cur["alerts"]))
    return {"new": len(cur["alerts"]) - persisted if prev else len(cur["alerts"]), "persisted": persisted, "gone": gone}


# ── 4. map clicks and heatmap ────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def nearest_alert(click: Optional[Dict], alerts: List[Dict], max_km: float = 1.5) -> Optional[Dict]:
    """The alert closest to a map click ({'lat':…, 'lng':…}), if within `max_km`; else None."""
    if not click or "lat" not in click or "lng" not in click or not alerts:
        return None
    best, best_d = None, max_km
    for a in alerts:
        d = haversine_km(click["lat"], click["lng"], a["lat"], a["lon"])
        if d <= best_d:
            best, best_d = a, d
    return best


def heat_points(alerts: Iterable[Dict]) -> List[List[float]]:
    """[lat, lon, weight 0..1] for a heat layer; weight blends peak radiance and confidence."""
    alerts = list(alerts)
    if not alerts:
        return []
    peak = max(float(a.get("delta_max", 0) or 0) for a in alerts) or 1.0
    return [[a["lat"], a["lon"],
             round(0.6 * min(1.0, float(a.get("delta_max", 0) or 0) / peak) + 0.4 * float(a.get("confidence", 0)) / 100, 3)]
            for a in alerts]


ZONE_STYLE = {   # fill colour for the boundary by how an alert relates to it (used for legend and tooltips)
    "official": "#22d3ee", "protected": "#34d399", "forest": "#86efac", "outside": "#64748b",
}


# ── 5. review board and activity feed ────────────────────────────────────────

BOARD_COLUMNS = [
    ("new", "New"), ("in review", "In review"), ("verified", "Verified"), ("false positive", "False positive"),
    ("closed", "Closed"),
]


def board_columns(statuses: Sequence[str]) -> List[Tuple[str, str]]:
    """Board columns restricted to statuses the store actually supports, in workflow order."""
    known = set(statuses)
    cols = [(k, v) for k, v in BOARD_COLUMNS if k in known]
    extras = [(s, s.capitalize()) for s in statuses if s not in {k for k, _ in BOARD_COLUMNS}]
    return cols + extras


def next_status(current: str, statuses: Sequence[str]) -> Optional[str]:
    """The next step in the workflow, or None at the end (verified → closed, etc.)."""
    flow = [k for k, _ in BOARD_COLUMNS if k in set(statuses) and k != "false positive"]
    if current in flow and flow.index(current) + 1 < len(flow):
        return flow[flow.index(current) + 1]
    return None


def group_by_status(alerts: Iterable[Dict], statuses: Sequence[str]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {s: [] for s, _ in board_columns(statuses)}
    for a in alerts:
        out.setdefault(a.get("status") or "new", []).append(a)
    for lst in out.values():
        lst.sort(key=lambda a: (-(a.get("confidence") or 0)))
    return out


def activity_feed(status_rows: Iterable[Dict], run_rows: Iterable[Dict], limit: int = 12) -> List[Dict]:
    """Merge review changes and scan runs into one newest-first list of {when, icon, text}."""
    items = []
    for r in status_rows:
        who = r.get("actor") or "someone"
        items.append({"when": str(r.get("changed_at", "")), "icon": "📝",
                      "text": f'{who} moved {r.get("alert_id")} from {r.get("old_status") or "new"} to {r.get("new_status")}'})
    for r in run_rows:
        ok = r.get("status") == "ok"
        items.append({"when": str(r.get("started_at", "")), "icon": "🛰️" if ok else "⚠️",
                      "text": (f'Scan of {r.get("region")}: {int(r.get("n_alerts") or 0)} alerts'
                               if ok else f'Scan of {r.get("region")} failed')})
    items.sort(key=lambda i: i["when"], reverse=True)
    return items[:limit]


# ── 6. live status strip ─────────────────────────────────────────────────────

def _parse_dt(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s)[:len(datetime.now().strftime(fmt))], fmt)
        except ValueError:
            continue
    return None


def ago(then: Optional[datetime], now: Optional[datetime] = None) -> str:
    if then is None:
        return "never"
    secs = int(((now or datetime.now()) - then).total_seconds())
    if secs < 90:
        return "just now"
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if secs >= size:
            n = secs // size
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return "just now"


def next_composite_estimate(period_label: str) -> Optional[date]:
    """
    Rough date of the next monthly VIIRS composite after the one labelled 'YYYY-MM'.
    Composites usually appear a few weeks after the month ends, so this is only an estimate.
    """
    try:
        y, m = int(period_label[:4]), int(period_label[5:7])
    except (ValueError, TypeError):
        return None
    m += 2                       # month after the next one starts; publication lags it by ~3 weeks
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return date(y, m, 1).replace(day=21)


def live_items(last_scan_at: str, period: str, last_run_ok: Optional[bool], now: Optional[datetime] = None
               ) -> List[Tuple[str, str, str]]:
    """(label, value, tone) chips for the status strip; tone is good / warn / info."""
    now = now or datetime.now()
    then = _parse_dt(last_scan_at) if last_scan_at else None
    items = [("Last scan", ago(then, now), "good" if then and (now - then).days < 35 else "warn")]
    nxt = next_composite_estimate(period or "")
    if nxt:
        days = (nxt - now.date()).days
        items.append(("Next satellite month", f"~{nxt:%d %b}" + (" (due)" if days <= 0 else f" · in {days}d"), "info"))
    if last_run_ok is not None:
        items.append(("Last run", "healthy" if last_run_ok else "failed", "good" if last_run_ok else "warn"))
    return items
