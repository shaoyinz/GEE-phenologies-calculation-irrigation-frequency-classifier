import ee
from shapely.wkt import loads
import pandas as pd

def initialize_ee():
    """Authenticates and initializes Earth Engine."""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

def convert_wkt_to_ee_geom(wkt_string):
    """Converts a WKT string to an Earth Engine Geometry."""
    try:
        # Handle cases where geometry might be NaN in the CSV
        if pd.isna(wkt_string):
            return None
        return ee.Geometry(loads(wkt_string).__geo_interface__)
    except Exception as e:
        print(f"Error converting geometry: {e}")
        return None