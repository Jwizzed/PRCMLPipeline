import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataCleaning:
    """
    Data cleaning class which preprocesses the data.
    """

    def __init__(self, contamination=0.01):
        self.contamination = contamination

    def clean_train_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans general dataframes (train and test data) using Isolation Forest."""
        logging.info("Cleaning DataFrame...")
        logging.info(f"Original shape: {df.shape}")
        cleaned_df = df.copy()
        cleaned_df = cleaned_df.dropna()

        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cleaned_df[numeric_columns])

        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(scaled_data)

        cleaned_df = cleaned_df[outlier_labels == 1].reset_index(drop=True)

        logging.info(f"Cleaned shape: {cleaned_df.shape}")
        return cleaned_df

    def clean_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans trajectory data using Isolation Forest per flight."""
        logging.info("Cleaning Trajectory DataFrame...")
        logging.info(f"Original shape: {df.shape}")
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

            iso_forest = IsolationForest(
                contamination=self.contamination, random_state=42
            )
            outlier_labels = iso_forest.fit_predict(scaled_data)

            return group[outlier_labels == 1]

        cleaned_df = cleaned_df.groupby("flight_id").apply(clean_group)

        cleaned_df = cleaned_df.reset_index(drop=True)
        logging.info(f"Cleaned shape: {cleaned_df.shape}")
        return cleaned_df
