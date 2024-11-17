import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataNormalizer:
    def __init__(self, exclude_columns: list, split_by_flown_distance: bool = False):
        self.exclude_columns = exclude_columns.copy()
        if split_by_flown_distance:
            self.exclude_columns.append("flown_distance")

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = df.copy()
        columns_to_normalize = df.select_dtypes(include=[np.number]).columns.difference(
            self.exclude_columns
        )

        scaler = StandardScaler()
        df_normalized[columns_to_normalize] = scaler.fit_transform(
            df[columns_to_normalize]
        )

        return df_normalized
