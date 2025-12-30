import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class IrrigationClassifier:
    def __init__(self, target_col='irr_group', n_estimators=80, random_state=42):
        self.target_col = target_col
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.top_features = []

    def prepare_data(self, df, feature_cols=None):
        """Encodes target and selects features."""
        # Target Encoding: '2' -> 0, '3-4' -> 1
        y = df[self.target_col].apply(lambda x: 0 if str(x) == '2' else 1)
        
        # Drop metadata columns to isolate features
        exclude = ['group_id', 'plot', 'sowing_group', 'irr_freq', 'irr_group', 'DOS_2022', 'date_of_image', 'irr_dates']
        X = df.drop(columns=[c for c in exclude if c in df.columns])
        
        # If specific features provided, filter them
        if feature_cols:
            X = X[feature_cols]
            
        return X, y

    def select_features(self, X, y, n_top=20):
        """Selects top features using Mutual Information."""
        print("Selecting features using Mutual Information...")
        mi = mutual_info_classif(X.fillna(0), y, random_state=42)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        self.top_features = mi_series.head(n_top).index.tolist()
        print(f"Top 5 Features: {self.top_features[:5]}")
        return self.top_features

    def train_evaluate(self, df, subset_name="All Data"):
        """Performs CV and prints report."""
        X, y = self.prepare_data(df)
        
        # Use selected features if available
        if self.top_features:
            X = X[self.top_features]
        
        X = X.fillna(0) # Simple imputation for RF

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"\n--- Evaluating Model: {subset_name} ---")
        
        # Cross Validate
        cv_results = cross_validate(self.model, X, y, cv=skf, scoring='accuracy')
        print(f"Mean CV Accuracy: {cv_results['test_score'].mean():.4f}")

        # Detailed Report
        y_pred = cross_val_predict(self.model, X, y, cv=skf)
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
        
        # Fit final model
        self.model.fit(X, y)
        return self.model