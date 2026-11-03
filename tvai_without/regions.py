"""
TerraVision AI — supported forest regions (plain Python, no Earth Engine needed).

Bounds are (min_lon, min_lat, max_lon, max_lat) and are the exact query boxes
used for the satellite data. gee_export.py, pipeline.py and app.py all read
from here so they can never drift apart.
"""

# Each region = a search box (used to download satellite rasters) + the countries whose land
# it covers. The box alone is NOT the region: several boxes spill into neighbouring countries
# and cities (the Aravalli box contains Jaipur, the Northeast India box reaches into Bangladesh,
# Myanmar, Bhutan and China). The real region is the designated protected areas and forest cover
# inside the box AND inside these countries — see boundaries.py.
#   bounds = (min_lon, min_lat, max_lon, max_lat)   iso3 = ISO 3166-1 alpha-3 codes
REGION_SPECS = {
    "Western Ghats, India":           {"bounds": (74.0, 8.0, 78.5, 15.5),    "iso3": ["IND"]},
    "Aravalli Hills, India":          {"bounds": (72.5, 23.5, 77.5, 28.5),   "iso3": ["IND"]},
    "Northeast India Forests":        {"bounds": (89.5, 21.5, 97.5, 29.0),   "iso3": ["IND"]},
    "Sundarbans, Bangladesh":         {"bounds": (88.5, 21.5, 90.0, 22.5),   "iso3": ["BGD"]},
    "Amazon Basin, Brazil":           {"bounds": (-73.0, -10.0, -44.0, 5.0), "iso3": ["BRA"]},
    "Atlantic Forest, Brazil":        {"bounds": (-50.0, -25.0, -35.0, -10.0), "iso3": ["BRA"]},
    "Tongass National Forest, USA":   {"bounds": (-136.0, 54.5, -129.0, 60.0), "iso3": ["USA"]},
    "Congo Basin, Africa":            {"bounds": (15.0, -5.0, 30.0, 5.0),
                                       "iso3": ["COD", "COG", "GAB", "CMR", "CAF", "GNQ"]},
    "Borneo Rainforest":              {"bounds": (108.0, -4.5, 119.0, 7.0),  "iso3": ["IDN", "MYS", "BRN"]},
    "Daintree Rainforest, Australia": {"bounds": (145.2, -16.5, 145.7, -15.8), "iso3": ["AUS"]},
    "Black Forest, Germany":          {"bounds": (7.5, 47.5, 8.5, 48.8),     "iso3": ["DEU"]},
}

# Kept for existing code: the search boxes
REGION_BOUNDS = {name: spec["bounds"] for name, spec in REGION_SPECS.items()}

# Map centre (lat, lon) for each region
REGION_COORDS = {
    "Western Ghats, India":           [12.0,   76.5],
    "Aravalli Hills, India":          [26.0,   74.5],
    "Northeast India Forests":        [25.0,   93.0],
    "Sundarbans, Bangladesh":         [21.9,   89.2],
    "Amazon Basin, Brazil":           [-3.5,  -60.0],
    "Atlantic Forest, Brazil":        [-18.0, -43.0],
    "Tongass National Forest, USA":   [57.0, -133.0],
    "Congo Basin, Africa":            [-1.0,   24.0],
    "Borneo Rainforest":              [1.5,   114.0],
    "Daintree Rainforest, Australia": [-16.1, 145.4],
    "Black Forest, Germany":          [48.2,    8.1],
}


def region_slug(name: str) -> str:
    """Same slug the original export/scan code used for file names."""
    return name.lower().replace(" ", "_").replace(",", "")
