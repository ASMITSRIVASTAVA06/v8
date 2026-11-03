import os
import ee
from dotenv import load_dotenv

load_dotenv()
ee.Authenticate()
ee.Initialize(project=os.getenv("GEE_PROJECT_ID"))

image = ee.Image("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG/20240101")
print("Connected! Bands:", image.bandNames().getInfo())