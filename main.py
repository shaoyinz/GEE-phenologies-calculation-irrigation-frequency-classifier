import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import custom modules
from gee_utils import initialize_ee
from downloaders import Sentinel2Downloader, LandsatThermalDownloader
from processors import TimeSeriesProcessor, FeatureEngineer
from visualizers import DataVisualizer
from models import IrrigationClassifier

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input Files
# Note: Ensure these files exist in your 'data' directory
INPUT_SHP = DATA_DIR / "field_data_treatment_with_groups.geojson"
INPUT_CSV = DATA_DIR / "field_data_treatment_with_groups.csv"

# Intermediate/Output Files
# Sentinel-2
S2_RAW = DATA_DIR / "sentinel_raw.csv"
S2_CLEANED = DATA_DIR / "sentinel_cleaned_resampled.csv"
# Thermal
THERMAL_RAW = DATA_DIR / "thermal_raw.csv"
THERMAL_CLEANED = DATA_DIR / "thermal_cleaned_resampled.csv"

# Features & Analysis Output
FEATURES_FILE = DATA_DIR / "features_engineered.csv"
PLOTS_FILE_S2 = DATA_DIR / "analysis_plots_indices.pdf"
PLOTS_FILE_THERMAL = DATA_DIR / "analysis_plots_thermal.pdf"

# Cleaning Parameters
# Biologically/Physically valid ranges for filtering
SENSOR_THRESHOLDS_S2 = {
    'ndvi': {'min': -0.2, 'max': 0.99},
    'gcvi': {'min': -1.0, 'max': 10.0},
    'ndmi': {'min': -1.0, 'max': 0.99},
    'cire': {'min': -1.0, 'max': 10.0}
}

SENSOR_THRESHOLDS_THERMAL = {
    'surface_temp_kelvin': {'min': 200, 'max': 350} # Approx range for Earth surface
}

# Analysis Filters
TARGET_VARIETY = 'HD2967'

def main():
    print("==================================================")
    print("       STARTING INTEGRATED ANALYSIS PIPELINE      ")
    print("==================================================")

    # ------------------------------------------------------
    # STEP 0: INITIALIZATION & METADATA LOADING
    # ------------------------------------------------------
    print("\n[Step 0] Initializing...")
    initialize_ee()

    # Load Metadata (GeoJSON) used for merging later
    if INPUT_SHP.exists():
        gdf_meta = gpd.read_file(INPUT_SHP)
        # Ensure 'plot' column is integer for accurate merging
        if 'plot' in gdf_meta.columns:
            gdf_meta['plot'] = gdf_meta['plot'].astype(int)
    else:
        raise FileNotFoundError(f"Metadata file {INPUT_SHP} not found. Please upload/place the file.")

    # ------------------------------------------------------
    # STEP 1: DATA DOWNLOADING (GEE)
    # ------------------------------------------------------
    print("\n[Step 1] Downloading Satellite Data...")

    # 1.A Sentinel-2
    if not S2_RAW.exists():
        print("  -> Downloading Sentinel-2 Data (Indices)...")
        # Downloads data based on irrigation dates +/- offset
        s2_downloader = Sentinel2Downloader(start_offset=30, end_offset=150)
        s2_downloader.process_plots(
            input_csv_path=str(INPUT_CSV),
            output_csv_path=str(S2_RAW),
            output_columns=['ndvi', 'gcvi', 'ndmi', 'cire']
        )
    else:
        print("  -> Sentinel-2 data found, skipping download.")

    # 1.B Landsat Thermal
    if not THERMAL_RAW.exists():
        print("  -> Downloading Landsat Thermal Data...")
        l8_downloader = LandsatThermalDownloader(start_offset=30, end_offset=150)
        l8_downloader.process_plots(
            input_csv_path=str(INPUT_CSV),
            output_csv_path=str(THERMAL_RAW),
            output_columns=['surface_temp_kelvin']
        )
    else:
        print("  -> Thermal data found, skipping download.")

    # ------------------------------------------------------
    # STEP 2: PREPROCESSING (Clean, Resample, Merge)
    # ------------------------------------------------------
    print("\n[Step 2] Preprocessing Time Series...")

    processor = TimeSeriesProcessor(date_col='date_of_image')

    def process_dataset(raw_path, thresholds, value_cols):
        """Helper to load, clean, resample, and merge metadata."""
        if not os.path.exists(raw_path):
            print(f"    ! Warning: {raw_path} not found.")
            return pd.DataFrame()

        df_raw = pd.read_csv(raw_path)

        # 1. Remove Sensor Errors
        df_clean = processor.remove_sensor_errors(df_raw, thresholds)

        # 2. Resample to Daily Frequency (Interpolates gaps)
        processor.value_cols = value_cols
        df_res = processor.process_time_series(df_clean, freq='D')

        # 3. Merge Metadata
        # We need specific columns for grouping and feature engineering
        # Exclude 'irr_freq' and 'irr_dates' as they already exist in df_res from raw data
        meta_cols = ['group_id', 'plot', 'DOS_2022', 'VarName_T1', 'Treatment Name']
        avail_cols = [c for c in meta_cols if c in gdf_meta.columns]

        # Ensure join keys match types
        df_res['plot'] = df_res['plot'].astype(int)

        merged = df_res.merge(gdf_meta[avail_cols], on=['group_id', 'plot'], how='left')
        return merged

    # 2.A Process Sentinel-2
    print("  -> Processing Sentinel-2...")
    df_s2 = process_dataset(S2_RAW, SENSOR_THRESHOLDS_S2, ['ndvi', 'gcvi', 'ndmi', 'cire'])
    if not df_s2.empty:
        # Filter for Target Variety
        df_s2 = df_s2[df_s2['VarName_T1'] == TARGET_VARIETY]
        df_s2.to_csv(S2_CLEANED, index=False)

    # 2.B Process Thermal
    print("  -> Processing Thermal...")
    df_thermal = process_dataset(THERMAL_RAW, SENSOR_THRESHOLDS_THERMAL, ['surface_temp_kelvin'])
    if not df_thermal.empty:
        # Filter for Target Variety
        df_thermal = df_thermal[df_thermal['VarName_T1'] == TARGET_VARIETY]
        df_thermal.to_csv(THERMAL_CLEANED, index=False)

    # ------------------------------------------------------
    # STEP 3: FEATURE ENGINEERING
    # ------------------------------------------------------
    print("\n[Step 3] Feature Engineering...")

    engineer = FeatureEngineer(sowing_col='DOS_2022', date_col='date_of_image')

    # We primarily generate features on Indices (S2) for the classification model
    if not df_s2.empty:
        # 3.A Growth Features (AUC, Max, Mean for whole season)
        df_growth = engineer.calculate_growth_features(df_s2, value_cols=['ndvi', 'gcvi', 'ndmi', 'cire'])

        # 3.B Period-specific Features (Slopes, Max in specific DAS windows)
        df_periods = engineer.calculate_period_stats(df_s2, value_cols=['ndvi', 'gcvi', 'ndmi', 'cire'])

        # Merge all features
        df_features = pd.merge(df_growth, df_periods, on=['group_id', 'plot'])

        df_features.to_csv(FEATURES_FILE, index=False)
        print(f"  -> Features saved to {FEATURES_FILE}")
    else:
        print("  ! Skipping feature engineering (no S2 data).")
        df_features = pd.DataFrame()

    # ------------------------------------------------------
    # STEP 4: VISUALIZATION
    # ------------------------------------------------------
    print("\n[Step 4] Generating Plots...")

    # Helper to add 'sowing_group' column required for plotting
    def prep_viz_data(df):
        return engineer._get_planting_metadata(df)

    # 4.A Plot Sentinel-2 Indices
    if not df_s2.empty:
        viz_s2 = DataVisualizer(output_pdf=str(PLOTS_FILE_S2))
        df_s2_viz = prep_viz_data(df_s2)
        viz_s2.plot_time_series_by_group(
            df_s2_viz,
            categories=['normal_planting', 'later_planting'],
            indices=['ndvi', 'gcvi', 'ndmi', 'cire'],
            group_col='sowing_group'
        )

    # 4.B Plot Thermal Data
    if not df_thermal.empty:
        viz_thermal = DataVisualizer(output_pdf=str(PLOTS_FILE_THERMAL))
        df_thermal_viz = prep_viz_data(df_thermal)
        viz_thermal.plot_time_series_by_group(
            df_thermal_viz,
            categories=['normal_planting', 'later_planting'],
            indices=['surface_temp_kelvin'],
            group_col='sowing_group'
        )

    # ------------------------------------------------------
    # STEP 5: MODELING (Random Forest)
    # ------------------------------------------------------
    print("\n[Step 5] Model Training & Evaluation...")

    if df_features.empty:
        print("  ! Skipping modeling (no features found).")
        return

    classifier = IrrigationClassifier(target_col='irr_group', n_estimators=100)

    # Split dataset by Sowing Group
    df_normal = df_features[df_features['sowing_group'] == 'normal_planting']
    df_later = df_features[df_features['sowing_group'] == 'later_planting']

    print("\n  >>> Model A: Normal Planting Group")
    if len(df_normal) > 10:
        X_n, y_n = classifier.prepare_data(df_normal)
        # Feature Selection
        top_feats_n = classifier.select_features(X_n, y_n, n_top=15)
        # Train & Evaluate
        classifier.train_evaluate(df_normal, subset_name="Normal Planting")
    else:
        print("  -> Not enough data for Normal Planting model.")

    print("\n  >>> Model B: Later Planting Group")
    if len(df_later) > 10:
        X_l, y_l = classifier.prepare_data(df_later)
        # Feature Selection
        top_feats_l = classifier.select_features(X_l, y_l, n_top=15)
        # Train & Evaluate
        classifier.train_evaluate(df_later, subset_name="Later Planting")
    else:
        print("  -> Not enough data for Later Planting model.")

    print("\n==================================================")
    print("           ANALYSIS PIPELINE COMPLETED            ")
    print("==================================================")

if __name__ == "__main__":
    main()
