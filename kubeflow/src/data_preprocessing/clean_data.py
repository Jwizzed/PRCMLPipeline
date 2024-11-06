import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    @staticmethod
    def clean_dataframe(df: pd.DataFrame,
                        contamination: float = 0.01) -> pd.DataFrame:
        """Cleans a dataframe using Isolation Forest for outlier detection."""
        cleaned_df = df.copy()
        cleaned_df = cleaned_df.dropna()

        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cleaned_df[numeric_columns])

        iso_forest = IsolationForest(contamination=contamination,
                                     random_state=42)
        outlier_labels = iso_forest.fit_predict(scaled_data)

        return cleaned_df[outlier_labels == 1].reset_index(drop=True)

    @staticmethod
    def clean_trajectory(df: pd.DataFrame,
                         contamination: float = 0.01) -> pd.DataFrame:
        """Cleans a trajectory dataframe using Isolation Forest for outlier detection."""
        cleaned_df = df.copy()

        cleaned_df = cleaned_df[
            (cleaned_df["altitude"] >= 0) & (cleaned_df["altitude"] <= 47000)
            ]
        cleaned_df = cleaned_df.dropna()

        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns

        def clean_group(group):
            if len(group) <= 1:
                return group

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(group[numeric_columns])

            iso_forest = IsolationForest(contamination=contamination,
                                         random_state=42)
            outlier_labels = iso_forest.fit_predict(scaled_data)

            return group[outlier_labels == 1]

        cleaned_df = cleaned_df.groupby("flight_id").apply(clean_group)
        return cleaned_df.reset_index(drop=True)
