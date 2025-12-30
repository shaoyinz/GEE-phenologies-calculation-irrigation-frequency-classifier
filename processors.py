import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.integrate import trapezoid

class TimeSeriesProcessor:
    def __init__(self, date_col='date_of_image', value_cols=None):
        self.date_col = date_col
        self.value_cols = value_cols if value_cols else ['ndvi', 'gcvi', 'ndmi', 'cire']

    def remove_sensor_errors(self, df, thresholds):
        """
        Step 1: Detects and removes rows where sensor values are biologically impossible
        or outside valid ranges.
        
        Args:
            df (pd.DataFrame): Raw data.
            thresholds (dict): e.g., {'ndvi': {'min': -0.2, 'max': 1.0}}
        """
        print("Running Sensor Error Detection...")
        df_clean = df.copy()
        
        # Create a mask initialized to True (Keep all)
        mask = pd.Series(True, index=df_clean.index)

        for col, rules in thresholds.items():
            if col not in df_clean.columns:
                continue
            
            # Ensure numeric
            col_vals = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Update mask based on min/max
            if 'min' in rules:
                mask &= (col_vals >= rules['min'])
            if 'max' in rules:
                mask &= (col_vals <= rules['max'])
            
            # Must not be NaN for the columns we are thresholding
            mask &= col_vals.notna()

        # Return only valid rows
        return df_clean[mask].reset_index(drop=True)

    def _detect_abrupt_changes(self, df, threshold=0.3):
        """Internal method: Spike/Drop detection on a single sorted partition."""
        df = df.copy().reset_index(drop=True)
        
        for col in self.value_cols:
            if col not in df.columns: 
                continue

            for i in range(1, len(df) - 1):
                prev_val = df.at[i-1, col]
                curr_val = df.at[i, col]
                next_val = df.at[i+1, col]
                
                if pd.isna(prev_val) or pd.isna(curr_val) or pd.isna(next_val):
                    continue

                diff_prev = curr_val - prev_val
                diff_next = curr_val - next_val

                # Detect Spike (up-up) or Drop (down-down)
                is_spike = (diff_prev > threshold) and (diff_next > threshold)
                is_drop = (diff_prev < -threshold) and (diff_next < -threshold)

                if is_spike or is_drop:
                    # Simple interpolation based on days distance could go here
                    # For simplicity, just averaging neighbors
                    df.at[i, col] = (prev_val + next_val) / 2
        return df

    def _resample_partition(self, df, freq='D'):
        """Internal method: Resample a single partition."""
        df = df.set_index(pd.to_datetime(df[self.date_col]))
        
        # Only resample available value columns
        cols = [c for c in self.value_cols if c in df.columns]
        if not cols:
            return df.reset_index()

        # Resample and interpolate
        resampled = df[cols].resample(freq).mean().interpolate(method='linear')
        return resampled.reset_index()

    def process_time_series(self, df, group_cols=['group_id', 'plot'], freq='D', change_threshold=0.3):
        """
        Main pipeline method.
        1. Groups data by plot.
        2. Detects abrupt changes per plot.
        3. Resamples per plot.
        4. Recombines into a single DataFrame.
        """
        print("Partitioning and Resampling Time Series...")
        if df.empty:
            return df

        # Ensure date column is datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        results = []
        grouped = df.groupby(group_cols)

        for name, group in grouped:
            group = group.sort_values(self.date_col)
            
            # A. Detect Spikes
            group_cleaned = self._detect_abrupt_changes(group, threshold=change_threshold)
            
            # B. Resample
            group_resampled = self._resample_partition(group_cleaned, freq=freq)
            
            # C. Restore Metadata (Group IDs)
            # name is a tuple of values corresponding to group_cols
            for col_name, col_val in zip(group_cols, name):
                group_resampled[col_name] = col_val
                
            # Restore other metadata (take first row from original)
            meta_cols = [c for c in df.columns if c not in self.value_cols + [self.date_col] + group_cols]
            for mc in meta_cols:
                # Use a safe get from the first row of original group
                try:
                    val = group.iloc[0][mc]
                    group_resampled[mc] = val
                except:
                    pass

            results.append(group_resampled)

        return pd.concat(results, ignore_index=True)
    
class FeatureEngineer:
    def __init__(self, sowing_col='DOS_2022', date_col='date_of_image'):
        self.sowing_col = sowing_col
        self.date_col = date_col

    def _get_planting_metadata(self, df):
        """Helper to create sowing groups and irrigation buckets."""
        df = df.copy()
        # Convert columns to datetime
        df[self.sowing_col] = pd.to_datetime(df[self.sowing_col])
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Define Sowing Groups
        df['sowing_group'] = df[self.sowing_col].apply(
            lambda x: 'normal_planting' if x <= pd.to_datetime('2022-11-30') else 'later_planting'
        )
        
        # Define Irrigation Groups (2 vs 3-4)
        if 'irr_freq' in df.columns:
            df['irr_group'] = df['irr_freq'].apply(
                lambda v: '2' if v == 2 else ('3-4' if v in [3, 4] else str(int(v)))
            )
        return df

    def calculate_growth_features(self, df, value_cols=['ndvi', 'gcvi', 'ndmi', 'cire'], days_buffer=20):
        """
        Calculates cumulative stats (Mean, Max, AUC) up to 105 + x days.
        """
        print("Calculating Growth Features (AUC, Max, Mean)...")
        df = self._get_planting_metadata(df)
        results = []
        
        # Group by plot identifier
        for (group_id, plot), group_data in df.groupby(['group_id', 'plot']):
            sowing_date = group_data[self.sowing_col].iloc[0]
            target_date = sowing_date + timedelta(days=105 + days_buffer)
            
            # Filter time window
            mask = (group_data[self.date_col] >= sowing_date) & (group_data[self.date_col] <= target_date)
            filtered = group_data.loc[mask].sort_values(self.date_col)
            
            if filtered.empty:
                continue

            row_stats = {
                'group_id': group_id,
                'plot': plot,
                'sowing_group': filtered['sowing_group'].iloc[0],
                'irr_group': filtered['irr_group'].iloc[0] if 'irr_group' in filtered else None
            }

            for col in value_cols:
                if col not in filtered.columns: continue
                
                valid_data = filtered.dropna(subset=[col])
                vals = valid_data[col].values
                days = (valid_data[self.date_col] - sowing_date).dt.days.values
                
                if len(vals) > 0:
                    row_stats[f'mean_{col}'] = vals.mean()
                    row_stats[f'max_{col}'] = vals.max()
                    row_stats[f'auc_{col}'] = trapezoid(vals, x=days) if len(vals) > 1 else 0
                    
                    # Days to max
                    max_idx = vals.argmax()
                    row_stats[f'days_to_max_{col}'] = days[max_idx]
                else:
                    row_stats[f'mean_{col}'] = 0
                    row_stats[f'max_{col}'] = 0
                    row_stats[f'auc_{col}'] = 0
            
            results.append(row_stats)
            
        return pd.DataFrame(results)

    def calculate_period_stats(self, df, value_cols=['ndvi'], periods=None):
        """
        Calculates Max and Slope (Rate of Change) for specific DAS windows.
        """
        print("Calculating Period-specific Stats...")
        if periods is None:
            periods = {
                '0_21_DAS': (0, 21),
                '21_65_DAS': (21, 65),
                '65_85_DAS': (65, 85),
                '85_105_DAS': (85, 105),
                '105_125_DAS': (105, 125)
            }
            
        df = self._get_planting_metadata(df)
        results = []

        for (group_id, plot), group_data in df.groupby(['group_id', 'plot']):
            sowing_date = group_data[self.sowing_col].iloc[0]
            
            row_stats = {'group_id': group_id, 'plot': plot}
            
            for col in value_cols:
                for p_name, (start, end) in periods.items():
                    p_start = sowing_date + timedelta(days=start)
                    p_end = sowing_date + timedelta(days=end)
                    
                    mask = (group_data[self.date_col] >= p_start) & (group_data[self.date_col] <= p_end)
                    slice_data = group_data.loc[mask, col]
                    
                    # MAX
                    row_stats[f'max_{col}_{p_name}'] = slice_data.max() if not slice_data.empty else 0
                    
                    # RATE OF CHANGE
                    if len(slice_data) > 1:
                        # Calculate numerical derivative (slope)
                        # Sorting is crucial
                        sorted_slice = group_data.loc[mask].sort_values(self.date_col)
                        vals = sorted_slice[col].values
                        
                        # Use simple difference of means or linear regression slope
                        # Here using the difference method from your notebook
                        diffs = np.diff(vals)
                        # Normalize by value magnitude to get rate? 
                        # Or just simple slope. Your notebook used: (diff / values).mean()
                        # We will replicate that logic safely:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            rates = diffs / vals[:-1] # diff is len-1
                            rates = rates[np.isfinite(rates)]
                            row_stats[f'change_rate_{col}_{p_name}'] = rates.mean() if len(rates) > 0 else 0
                    else:
                        row_stats[f'change_rate_{col}_{p_name}'] = 0

            results.append(row_stats)

        return pd.DataFrame(results)