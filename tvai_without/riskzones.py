"""
TerraVision AI — risk zones: weather forecast + cyclone tracks combined per map cell and per region.

`assess_bounds` returns grid cells with a combined level (the worse of forecast hazards and cyclone
proximity); `region_overview` does the same coarsely for every forest region so a whole network of
forests can be watched at a glance; `zone_label` names a cell for tables and alerts.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cyclones
import weather
from regions import REGION_BOUNDS


def combine(cells: List[Dict], storms: List[Dict]) -> List[Dict]:
    """Add cyclone proximity to each cell (in place) and raise its level if the storm is the bigger threat."""
    for c in cells:
        t = cyclones.worst_cyclone_threat(c["lat"], c["lon"], storms) if storms else None
        if t and t["level"] > 0:
            c["hazards"]["cyclone"] = {"level": t["level"], "value": t["distance_km"], "unit": f"km from {t['storm']}",
                                       "at": (t["nearest"] or {}).get("time", ""), "reason": t["reason"]}
            c["level"] = max(c["level"], t["level"])
            c["headline"] = weather.headline(c["hazards"])
    return cells


def assess_bounds(bounds: Tuple[float, float, float, float], storms: Optional[List[Dict]] = None,
                  step_deg: float = 1.0, hours: int = 72,
                  http: Optional[Callable[[str], object]] = None) -> Dict:
    area = weather.assess_area(bounds, step_deg, hours, http)
    combine(area["cells"], storms or [])
    area["worst"] = max((c["level"] for c in area["cells"]), default=0)
    return area


def zone_label(cell: Dict) -> str:
    return f"{cell['lat']:.2f}°, {cell['lon']:.2f}°"


def area_row(name: str, area: Dict) -> Dict:
    """Summary row for one assessed area (used for the overview table and for alerts)."""
    worst = max(area["cells"], key=lambda c: c["level"], default=None)
    peak = ""
    if worst:
        peak = next((h["at"] for h in worst["hazards"].values() if h["level"] == worst["level"] and h.get("at")), "")
    return {"region": name, "level": area["worst"] if area["cells"] else None,
            "headline": worst["headline"] if worst else "No forecast available",
            "flagged": sum(1 for c in area["cells"] if c["level"] >= 2), "cells": len(area["cells"]),
            "peak_at": peak, "error": "; ".join(area["errors"])[:120]}


def region_overview(storms: Optional[List[Dict]] = None, hours: int = 72, step_deg: float = 2.0,
                    regions: Optional[List[str]] = None,
                    http: Optional[Callable[[str], object]] = None) -> List[Dict]:
    """One row per forest region: worst level, what the hazard is, how many cells are flagged, failures."""
    rows = []
    for name in regions or list(REGION_BOUNDS):
        try:
            rows.append(area_row(name, assess_bounds(REGION_BOUNDS[name], storms, step_deg, hours, http)))
        except Exception as e:                              # one region failing must not hide the others
            rows.append({"region": name, "level": None, "headline": f"Could not check ({type(e).__name__})",
                         "flagged": 0, "cells": 0, "peak_at": "", "error": str(e)[:120]})
    return sorted(rows, key=lambda r: -(r["level"] if r["level"] is not None else -1))
