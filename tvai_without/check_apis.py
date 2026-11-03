"""
TerraVision AI — check that every data source works with your keys.

  python check_apis.py

Makes one tiny request per service and prints OK / FAILED with a plain reason. Keys are never printed.
Run this once after editing .env, and again whenever something looks wrong.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional

import apis


def _check_firms(http: Optional[Callable[[str], str]] = None) -> str:
    import firms
    pts = firms.fetch_hotspots((77.0, 27.0, 78.0, 28.0), days=1, http=http)
    return f"key accepted ({len(pts)} hotspot(s) in a test box over the last day)"


def _check_weather(http=None) -> str:
    import weather
    out, errs = weather.fetch_forecast([{"lat": 27.0, "lon": 77.0}], 24, http)
    if out[0] is None:
        raise RuntimeError(errs[0] if errs else "no data")
    return f"forecast received ({len(out[0].get('time', []))} hourly steps) from {apis.weather_endpoint()['url'].split('/')[2]}"


def _check_geocoder(http=None) -> str:
    import geocode
    r = geocode.search_place("Jaipur, India", limit=1)
    if not r:
        raise RuntimeError("no result for a known city")
    return f"found {r[0]['name'][:40]}"


def _check_cyclones(http=None) -> str:
    import cyclones
    storms, _ = cyclones.list_active_cyclones(with_tracks=False)
    return f"feed reachable ({len(storms)} active cyclone(s))"


def _check_gee(http=None) -> str:
    from gee_gateway import connect_gee
    connect_gee()
    return "authenticated"


CHECKS: Dict[str, Callable] = {"firms": _check_firms, "weather": _check_weather, "geocoder": _check_geocoder,
                               "cyclones": _check_cyclones, "gee": _check_gee}


def run(checks: Optional[Dict[str, Callable]] = None) -> List[Dict]:
    results = []
    for row in apis.status_rows():
        fn = (checks or CHECKS).get(row["id"])
        if row["state"] == "missing" or fn is None:
            results.append({**row, "ok": None, "message": row["detail"]})
            continue
        try:
            results.append({**row, "ok": True, "message": fn(None)})
        except Exception as e:      # report, never crash: one broken key must not hide the others
            results.append({**row, "ok": False, "message": apis.redact(f"{type(e).__name__}: {str(e)[:160]}")})
    return results


def main() -> int:
    bad = 0
    for r in run():
        tag = {True: "OK     ", False: "FAILED ", None: "SKIPPED"}[r["ok"]]
        print(f"{tag} {r['name']:<32} {r['message']}")
        bad += r["ok"] is False
    print("\nKeys are never printed by this tool.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
