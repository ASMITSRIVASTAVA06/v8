"""
TerraVision AI — alert tracking across scans.

Matches each alert in a new scan against the previous scan of the same region
by distance, so a cluster that keeps glowing keeps its ID, and we can tell
which alerts are NEW, which are RECURRING, and which have RESOLVED.
Pure NumPy — no Earth Engine or Streamlit needed.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0088
DEFAULT_MATCH_KM = 1.0   # two VIIRS pixels; a persistent source drifts less than this


def new_alert_id(region: str, lat: float, lon: float, category: str) -> str:
    raw = f"{region}|{lat:.4f}|{lon:.4f}|{category}".encode()
    return "TV-" + hashlib.sha1(raw).hexdigest()[:8].upper()


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Pairwise great-circle distance in km, shape (len(1), len(2))."""
    p1, p2 = np.radians(np.asarray(lat1))[:, None], np.radians(np.asarray(lat2))[None, :]
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2))[None, :] - np.radians(np.asarray(lon1))[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_alerts(current: List[Dict], previous: List[Dict],
                 max_km: float = DEFAULT_MATCH_KM) -> Dict[int, int]:
    """
    One-to-one nearest matching. Returns {index_in_current: index_in_previous}.
    Closest pairs are assigned first, so two nearby clusters cannot both claim
    the same predecessor.
    """
    if not current or not previous:
        return {}
    d = haversine_matrix(
        [a["lat"] for a in current], [a["lon"] for a in current],
        [a["lat"] for a in previous], [a["lon"] for a in previous],
    )
    ci, pi = np.where(d <= max_km)
    order = np.argsort(d[ci, pi])
    used_c, used_p, out = set(), set(), {}
    for k in order:
        c, p = int(ci[k]), int(pi[k])
        if c in used_c or p in used_p:
            continue
        used_c.add(c); used_p.add(p); out[c] = p
    return out


def apply_tracking(
    region: str,
    current: List[Dict],
    previous: List[Dict],
    scanned_at: str,
    period: str,
    previous_period: str | None,
    comparable: bool = True,
    max_km: float = DEFAULT_MATCH_KM,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Annotate `current` alerts (in place) with alert_id, lifecycle, first_seen and
    times_seen. Returns (current, resolved) where `resolved` are previous alerts
    that no longer appear.

    `comparable` should be False when the previous scan used different detection
    settings: matching still carries IDs over, but nothing is reported as
    resolved because a stricter threshold would make everything look "resolved".
    A re-scan of the same satellite period never counts as new evidence, so
    times_seen only grows when the period changed.
    """
    matches = match_alerts(current, previous, max_km)
    same_period = previous_period is not None and previous_period == period

    for i, a in enumerate(current):
        if i in matches:
            prev = previous[matches[i]]
            seen = int(prev.get("times_seen", 1))
            a["alert_id"] = prev.get("alert_id") or new_alert_id(region, a["lat"], a["lon"], a["category"])
            a["times_seen"] = seen if same_period else seen + 1
            a["first_seen"] = prev.get("first_seen") or prev.get("detected_at") or scanned_at
            a["lifecycle"] = "recurring"
        else:
            a["alert_id"] = new_alert_id(region, a["lat"], a["lon"], a["category"])
            a["times_seen"] = 1
            a["first_seen"] = scanned_at
            a["lifecycle"] = "new"

    # Rare: two different new alerts rounding to the same hash -> disambiguate
    seen_ids: Dict[str, int] = {}
    for a in current:
        n = seen_ids.get(a["alert_id"], 0)
        seen_ids[a["alert_id"]] = n + 1
        if n:
            a["alert_id"] = f"{a['alert_id']}-{n + 1}"

    resolved: List[Dict] = []
    if comparable and previous:
        matched_prev = set(matches.values())
        resolved = [
            {k: p.get(k) for k in ("alert_id", "lat", "lon", "category", "severity",
                                   "confidence", "first_seen", "times_seen")}
            for j, p in enumerate(previous) if j not in matched_prev
        ]
    return current, resolved
