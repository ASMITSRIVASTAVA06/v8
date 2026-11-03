"""
TerraVision AI — external data sources and their API keys.

One place that knows which services the app can use, where each key comes from, and whether it is
set. Keys are read from environment variables (your `.env` file) or, when running in Streamlit,
from `st.secrets`. A key is NEVER printed, logged, or put in an error message: `redact()` strips it
from any text before it can be shown.

  python check_apis.py        # tests every configured service with one tiny request
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:      # python-dotenv is optional
    pass

# id, name, what it gives the app, env variables, key needed?, where to get one, plan note
PROVIDERS: List[Dict] = [
    {"id": "firms", "name": "NASA FIRMS (active fire)", "env": ["FIRMS_MAP_KEY"], "needs_key": True,
     "purpose": "Real thermal hotspots within hours of the satellite pass: place search and fire checks.",
     "signup": "https://firms.modaps.eosdis.nasa.gov/api/map_key/", "cost": "Free"},
    {"id": "weather", "name": "Open-Meteo (weather forecast)", "env": ["OPENMETEO_API_KEY"], "needs_key": False,
     "purpose": "Hourly rain, gusts, heat and storm energy for the weather risk zones.",
     "signup": "https://open-meteo.com/en/pricing",
     "cost": "Free without a key for non-commercial use; a paid key is required for commercial use"},
    {"id": "gee", "name": "Google Earth Engine", "env": ["GEE_PROJECT_ID"], "needs_key": True,
     "purpose": "Night lights (VIIRS), protected areas, land cover, forest loss, vegetation.",
     "signup": "https://earthengine.google.com/", "cost": "Free for non-commercial / research use"},
    {"id": "geocoder", "name": "Geocoder (place search)", "env": ["GEOCODER_API_KEY"], "needs_key": False,
     "purpose": "Turns place names into coordinates. The free OpenStreetMap service is for light use only.",
     "signup": "https://locationiq.com/", "cost": "Free tier available"},
    {"id": "cyclones", "name": "GDACS (cyclone tracks)", "env": [], "needs_key": False,
     "purpose": "Active tropical cyclones with observed and forecast tracks.",
     "signup": "https://www.gdacs.org/", "cost": "Free, no key"},
]


def get_setting(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit secrets (if running inside Streamlit), else `default`."""
    val = os.environ.get(name)
    if val:
        return val.strip()
    try:
        import streamlit as st

        v = st.secrets.get(name)          # raises if no secrets file exists
        if v:
            return str(v).strip()
    except Exception:
        pass
    return default


def redact(text: object, extra: Optional[List[str]] = None) -> str:
    """`text` with every configured API key / token replaced by ***, safe to show or log."""
    out = str(text)
    secrets = [get_setting(e) for p in PROVIDERS for e in p["env"]] + [get_setting(k) for k in (
        "SMTP_PASSWORD", "TELEGRAM_BOT_TOKEN", "GEE_SERVICE_ACCOUNT_JSON", "GEE_CREDENTIALS_JSON")] + list(extra or [])
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "***")
    return out


def provider_status(p: Dict) -> Dict:
    """{id, name, state, detail}. state: live (key set) | free (works without a key) | missing (needs a key)."""
    have = [e for e in p["env"] if get_setting(e)]
    if p["id"] == "gee":
        have = have or ([e for e in ("GEE_SERVICE_ACCOUNT_JSON", "GEE_CREDENTIALS_JSON") if get_setting(e)])
    if p["id"] == "geocoder" and have and not get_setting("GEOCODER_URL"):
        return {"id": p["id"], "name": p["name"], "state": "missing", "purpose": p["purpose"], "signup": p["signup"],
                "cost": p["cost"], "detail": "key set but GEOCODER_URL is empty: add your provider's search URL"}
    if have:
        state, detail = "live", "key set"
    elif p["needs_key"]:
        state, detail = "missing", "add " + (p["env"][0] if p["env"] else "a key") + " to .env"
    else:
        state, detail = "free", "no key set: using the free service" if p["env"] else "no key needed"
    return {"id": p["id"], "name": p["name"], "state": state, "detail": detail, "purpose": p["purpose"],
            "signup": p["signup"], "cost": p["cost"]}


def status_rows() -> List[Dict]:
    return [provider_status(p) for p in PROVIDERS]


def firms_key() -> str:
    return get_setting("FIRMS_MAP_KEY")


def weather_endpoint() -> Dict:
    """Base URL and key parameter for the forecast: paid host when a key is set (override with OPENMETEO_BASE_URL)."""
    key = get_setting("OPENMETEO_API_KEY")
    base = get_setting("OPENMETEO_BASE_URL") or ("https://customer-api.open-meteo.com/v1/forecast" if key
                                                  else "https://api.open-meteo.com/v1/forecast")
    return {"url": base, "key": key}


def geocoder_endpoint() -> Dict:
    """
    Search URL and key. Any Nominatim-compatible service works (for example LocationIQ).
    The key is only ever sent to the URL in GEOCODER_URL: with no GEOCODER_URL the free OpenStreetMap
    service is used and the key is NOT attached to the request, so it can never leak to the wrong host.
    """
    url = get_setting("GEOCODER_URL")
    return {"url": url or "https://nominatim.openstreetmap.org/search",
            "key": get_setting("GEOCODER_API_KEY") if url else ""}
