import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEncoder:
    def __init__(self, preserve_columns: list):
        self.preserve_columns = preserve_columns
        self.categorical_col = [
            "adep",
            "country_code_adep",
            "ades",
            "country_code_ades",
            "aircraft_type",
            "airline",
        ]
        self.oneHot_col = ["wtc"]
        self.encoder = LabelEncoder()

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()

        # Label encoding
        for col in self.categorical_col:
            df_encoded[col + "_encoded"] = self.encoder.fit_transform(
                df_encoded[col])
            if col not in self.preserve_columns:
                df_encoded = df_encoded.drop(columns=[col])

        # One-hot encoding
        df_encoded = pd.get_dummies(df_encoded, columns=self.oneHot_col)

        # Convert to int
        df_encoded["wtc_M"] = df_encoded["wtc_M"].astype(int)
        df_encoded["wtc_H"] = df_encoded["wtc_H"].astype(int)

        return df_encoded
