import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast
from matplotlib.backends.backend_pdf import PdfPages

class DataVisualizer:
    def __init__(self, output_pdf='plots.pdf'):
        self.output_pdf = output_pdf

    def _parse_irr_dates(self, date_str):
        try:
            return [pd.to_datetime(d) for d in ast.literal_eval(date_str)]
        except:
            return []

    def plot_time_series_by_group(self, df, categories, indices, x_col='date_of_image', 
                                  hue_col='irr_group', group_col='sowing_group'):
        """
        Generates line plots with error bars for specified categories and indices.
        """
        print(f"Generating plots into {self.output_pdf}...")
        
        # Ensure dates are datetime
        df = df.copy()
        df[x_col] = pd.to_datetime(df[x_col])
        if 'DOS_2022' in df.columns:
            df['DOS_2022'] = pd.to_datetime(df['DOS_2022'])
        
        if 'irr_dates' in df.columns and isinstance(df['irr_dates'].iloc[0], str):
            df['irr_dates'] = df['irr_dates'].apply(self._parse_irr_dates)

        with PdfPages(self.output_pdf) as pdf:
            for cat in categories:
                cat_data = df[df[group_col] == cat]
                if cat_data.empty: continue

                for idx in indices:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Main Line Plot
                    sns.lineplot(
                        data=cat_data, x=x_col, y=idx, hue=hue_col,
                        palette='viridis', marker='o', errorbar='se', ax=ax
                    )
                    
                    # Background: Sowing Window
                    if 'DOS_2022' in df.columns:
                        dos_min = cat_data['DOS_2022'].min()
                        dos_max = cat_data['DOS_2022'].max()
                        ax.axvspan(dos_min, dos_max, color='gray', alpha=0.2, label='Sowing')

                    # Background: Irrigation Windows (Approximate)
                    colors = ['#ADD8E6', '#87CEEB', '#6495ED', '#4169E1']
                    if 'irr_dates' in cat_data.columns:
                        for i in range(4):
                            # Get all dates for the i-th irrigation event across the group
                            # This is tricky in pandas, iterating simply for display
                            try:
                                dates = cat_data['irr_dates'].apply(lambda x: x[i] if len(x) > i else pd.NaT).dropna()
                                if not dates.empty:
                                    ax.axvspan(dates.min(), dates.max(), color=colors[i], alpha=0.3)
                            except:
                                pass

                    ax.set_title(f'{idx.upper()} for {cat}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    pdf.savefig(fig)
                    plt.close(fig)
        print("Plots saved.")