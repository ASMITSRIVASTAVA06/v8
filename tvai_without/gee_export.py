import ee
from gee_gateway import connect_gee
from regions import REGION_BOUNDS, region_slug

# Connect to Earth Engine using the project ID from the .env file
connect_gee()



# Single source of truth for region boxes lives in regions.py
FOREST_REGIONS = {name: ee.Geometry.Rectangle(list(b)) for name, b in REGION_BOUNDS.items()}

# =============================================================================
# EXPORT FUNCTION
# Pulls baseline and current VIIRS data for a selected region and
# exports both as GeoTIFF files to Google Drive.
# =============================================================================

def export_region(region_name: str):
    """
    Export baseline (5-year median) and current VIIRS radiance rasters
    for the named region to Google Drive as GeoTIFF files.

    Files are named:
      viirs_baseline_{region_slug}.tif
      viirs_current_{region_slug}.tif
    """
    if region_name not in FOREST_REGIONS:
        raise ValueError(
            f"Region '{region_name}' not found. "
            f"Available: {list(FOREST_REGIONS.keys())}"
        )

    forest = FOREST_REGIONS[region_name]

    # Create a safe filename slug from the region name
    slug = region_slug(region_name)

    # 5-year median baseline (2019-2023)
    # Using median rather than mean makes the baseline robust to
    # occasional wildfire seasons or cloud contamination in any single year
    baseline = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate("2019-01-01", "2023-12-31")
        .filterBounds(forest)
        .select("avg_rad")
        .median()
    )

    # Current period: the newest available monthly composites (same rolling window the app uses)
    from pipeline import latest_viirs_date, rolling_window
    start, end, _label = rolling_window(latest_viirs_date(ee, REGION_BOUNDS[region_name]))
    current = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(forest)
        .select("avg_rad")
        .mean()
    )

    # Export baseline raster to Google Drive
    ee.batch.Export.image.toDrive(
        image=baseline,
        description=f"viirs_baseline_{slug}",
        region=forest,
        scale=500,
        fileFormat="GeoTIFF",
        maxPixels=1e10,
    ).start()

    # Export current raster to Google Drive
    ee.batch.Export.image.toDrive(
        image=current,
        description=f"viirs_current_{slug}",
        region=forest,
        scale=500,
        fileFormat="GeoTIFF",
        maxPixels=1e10,
    ).start()

    print(f"Export started for: {region_name}")
    print(f"  Baseline file: viirs_baseline_{slug}.tif")
    print(f"  Current file : viirs_current_{slug}.tif")
    print("Check Google Drive in 10-15 minutes.")


if __name__ == "__main__":
    # Change this to export a different region
    # export_region("Western Ghats, India")
    # export_region("Aravalli Hills, India")
    export_region("Amazon Basin, Brazil")