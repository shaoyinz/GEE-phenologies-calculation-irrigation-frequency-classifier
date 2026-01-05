import pandas as pd
import ee
import datetime
import ast
import csv
import gc
from abc import ABC, abstractmethod
from gee_utils import convert_wkt_to_ee_geom

class BaseGEEDownloader(ABC):
    """
    Abstract Base Class for downloading plot data from GEE.
    Handles the CSV streaming IO and iterating through plots.
    """
    def __init__(self, start_offset=30, end_offset=150):
        self.start_offset = start_offset
        self.end_offset = end_offset

    def _get_date_range(self, irr_dates_str):
        """Parses irrigation dates and calculates the study window."""
        try:
            irr_dates = ast.literal_eval(irr_dates_str)
            first_irr_date = datetime.datetime.strptime(irr_dates[0], '%Y-%m-%d').date()
            start_date = (first_irr_date - datetime.timedelta(days=self.start_offset)).strftime('%Y-%m-%d')
            end_date = (first_irr_date + datetime.timedelta(days=self.end_offset)).strftime('%Y-%m-%d')
            return start_date, end_date
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return None, None

    @abstractmethod
    def get_collection(self, start_date, end_date, geometry):
        """Child classes must implement specific image collection retrieval."""
        pass

    @abstractmethod
    def process_image_collection(self, collection):
        """Child classes must implement masking, indices, and mosaicking."""
        pass

    @abstractmethod
    def compute_stats(self, image, geometry):
        """Child classes must implement reduction/stats logic."""
        pass

    def process_plots(self, input_csv_path, output_csv_path, output_columns):
        """
        Main driver: Reads input CSV, loops through polygons, delegates GEE logic,
        and streams output row-by-row.
        """
        gdf = pd.read_csv(input_csv_path)
        
        # Open output file for streaming
        with open(output_csv_path, 'w', newline='') as f:
            # Combine metadata columns with dynamic data columns
            meta_columns = [col for col in gdf.columns if col not in ['geometry']]
            final_columns = ['date_of_image'] + output_columns + meta_columns
            
            writer = csv.DictWriter(f, fieldnames=final_columns)
            writer.writeheader()

            for i, row in gdf.iterrows():
                print(f"Processing field {i}")
                
                # 1. Parse Geometry
                polygon = convert_wkt_to_ee_geom(row['geometry'])
                if not polygon:
                    print(f"Skipping field {i}: Invalid geometry.")
                    continue

                # 2. Parse Dates
                start_date, end_date = self._get_date_range(row['irr_dates'])
                if not start_date:
                    continue
                print(f"Date Range: {start_date} to {end_date}")

                # 3. Get Collection
                collection = self.get_collection(start_date, end_date, polygon)
                if collection.size().getInfo() == 0:
                    print("No images found.")
                    continue

                # 4. Process (Masking, Indices, Mosaics)
                processed_collection = self.process_image_collection(collection, polygon)
                if processed_collection.size().getInfo() == 0:
                     print("No valid mosaics created.")
                     continue
                
                # 5. Compute Stats (Map over collection)
                # We wrap the compute_stats in a lambda to pass the specific polygon
                stats_fc = processed_collection.map(lambda img: self.compute_stats(img, polygon))
                
                # 6. Extract to Pandas
                try:
                    df_stats = ee.data.computeFeatures({'expression': stats_fc, 'fileFormat': 'PANDAS_DATAFRAME'})
                except Exception as e:
                    print(f"GEE Compute Error: {e}")
                    continue

                if df_stats.empty:
                    print("No data returned.")
                    continue

                df_stats['date_of_image'] = pd.to_datetime(df_stats['millis'], unit='ms').dt.date

                # 7. Write to CSV
                for _, stat_row in df_stats.iterrows():
                    final_row = row.to_dict() # Metadata
                    final_row.update(stat_row.to_dict()) # Satellite data
                    # Filter to only ensure we write columns defined in header
                    writer.writerow({k: final_row.get(k) for k in final_columns})

                del df_stats
                gc.collect()
        
        print(f"\nProcessing complete. Saved to {output_csv_path}")


class Sentinel2Downloader(BaseGEEDownloader):
    def get_collection(self, start_date, end_date, geometry):
        return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                 .filterBounds(geometry)\
                 .filterDate(start_date, end_date)

    def process_image_collection(self, collection, geometry):
        # Define cloud masking and indices logic here (condensed for brevity)
        def mask_edges(image):
            return image.updateMask(image.select('B8A').mask().updateMask(image.select('B9').mask()))

        def add_indices(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
            gcvi = image.expression('(NIR / GREEN) - 1', {'NIR': image.select('B8'), 'GREEN': image.select('B3')}).rename('gcvi')
            ndmi = image.normalizedDifference(['B8A', 'B11']).rename('ndmi')
            cire = image.expression('(NIR / RedEdge1) - 1', {'NIR': image.select('B8'), 'RedEdge1': image.select('B5')}).rename('cire')
            return image.addBands([ndvi, gcvi, ndmi, cire])
            
        # Cloud masking using SCL band (Scene Classification Layer)
        def cloud_mask_s2(image):
            """Masks clouds and cloud shadows using the SCL band."""
            scl = image.select('SCL')
            # SCL values: 3=cloud shadows, 8=cloud medium probability, 9=cloud high probability, 10=thin cirrus
            mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            return image.updateMask(mask)
        
        processed = collection.map(mask_edges).map(add_indices).map(cloud_mask_s2)
        
        # Mosaic by date logic
        def mosaic_by_date(col):
            unique_dates = col.aggregate_array('system:time_start').map(
                lambda time_start: ee.Date(time_start).format('YYYY-MM-dd')
            ).distinct()

            def create_daily_mosaic(date_str):
                date = ee.Date(date_str)
                daily_collection = col.filterDate(date, date.advance(1, 'day'))

                # For multiple images on same day, use mosaic (takes first valid pixel)
                # This automatically handles overlap between tiles
                mosaic = daily_collection.mosaic()

                return mosaic.set('system:time_start', date.millis())

            daily_mosaics = ee.ImageCollection.fromImages(
                unique_dates.map(create_daily_mosaic)
            )
            return daily_mosaics

        return mosaic_by_date(processed)

    def compute_stats(self, image, geometry):
        reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
        stats = image.select(['ndvi', 'gcvi', 'ndmi', 'cire']).reduceRegion(
            reducer=reducer, geometry=geometry, scale=10, bestEffort=True
        )
        # Conditional logic for validity
        ndvi_count = ee.Number(stats.get('ndvi_count'))
        is_valid = ndvi_count.gt(0)
        
        return ee.Feature(None, {
            'millis': image.date().millis(),
            'ndvi': ee.Algorithms.If(is_valid, stats.get('ndvi_mean'), None),
            'gcvi': ee.Algorithms.If(is_valid, stats.get('gcvi_mean'), None),
            'ndmi': ee.Algorithms.If(is_valid, stats.get('ndmi_mean'), None),
            'cire': ee.Algorithms.If(is_valid, stats.get('cire_mean'), None),
        })


class LandsatThermalDownloader(BaseGEEDownloader):
    def get_collection(self, start_date, end_date, geometry):
        return ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")\
                 .filterBounds(geometry)\
                 .filterDate(start_date, end_date)

    def process_image_collection(self, collection, geometry):
        def apply_scale(image):
            thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
            return image.addBands(thermal.rename('ST_B10_K'), overwrite=True)

        def qa_mask(image):
            """Masks clouds, cloud shadows, and snow using QA_PIXEL band."""
            qa = image.select('QA_PIXEL')
            # Bit 3: Cloud
            # Bit 4: Cloud Shadow
            # Bit 5: Snow
            # Create mask where these bits are NOT set (0)
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)
            shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)
            snow_mask = qa.bitwiseAnd(1 << 5).eq(0)
            mask = cloud_mask.And(shadow_mask).And(snow_mask)
            return image.updateMask(mask)

        processed = collection.map(apply_scale).map(qa_mask)

        # Mosaic by date logic
        def mosaic_by_date(col):
            unique_dates = col.aggregate_array('system:time_start').map(
                lambda time_start: ee.Date(time_start).format('YYYY-MM-dd')
            ).distinct()

            def create_daily_mosaic(date_str):
                date = ee.Date(date_str)
                daily_collection = col.filterDate(date, date.advance(1, 'day'))

                # For multiple images on same day, use mosaic (takes first valid pixel)
                mosaic = daily_collection.mosaic()

                return mosaic.set('system:time_start', date.millis())

            daily_mosaics = ee.ImageCollection.fromImages(
                unique_dates.map(create_daily_mosaic)
            )
            return daily_mosaics

        return mosaic_by_date(processed)

    def compute_stats(self, image, geometry):
        stats = image.select(['ST_B10_K']).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=30, bestEffort=True
        )
        return ee.Feature(None, {
            'millis': image.date().millis(),
            'surface_temp_kelvin': stats.get('ST_B10_K')
        })