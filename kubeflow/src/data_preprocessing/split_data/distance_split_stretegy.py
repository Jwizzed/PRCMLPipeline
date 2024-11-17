from typing import Dict, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kubeflow.src.data_preprocessing.split_data.split_data_strategy import (
    SplitDataStrategy,
)


class DistanceSplitStrategy(SplitDataStrategy):
    def process_subset(self, subset_df: pd.DataFrame, is_test: bool = False):
        subset_df = subset_df.copy()
        scaler = StandardScaler()
        subset_df[["MTOW(kg)", "flown_distance"]] = scaler.fit_transform(
            subset_df[["MTOW(kg)", "flown_distance"]]
        )

        X = subset_df.drop(["tow", "flown_distance_range"], axis=1).reset_index(
            drop=True
        )
        y = (
            subset_df[["flight_id", "tow"]]
            .drop("flight_id", axis=1)
            .reset_index(drop=True)
        )
        return X, y

    def split(
        self, df: pd.DataFrame, **kwargs
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        narrow_body_df = df[df["MTOW(kg)"] <= 115680].copy()
        wide_body_df = df[df["MTOW(kg)"] > 115680].copy()

        bins = [0, 500, 1000, float("inf")]
        labels = ["0-500", "500-1000", ">1000"]
        narrow_body_df["flown_distance_range"] = pd.cut(
            narrow_body_df["flown_distance"], bins=bins, labels=labels, right=False
        )

        wide_body_df["flown_distance_range"] = wide_body_df["aircraft_type"].apply(
            lambda x: "B77W" if x == "B77W" else "NonB77W"
        )

        result = {
            "0-500": self.process_subset(
                narrow_body_df[narrow_body_df["flown_distance_range"] == "0-500"]
            ),
            "500-1000": self.process_subset(
                narrow_body_df[narrow_body_df["flown_distance_range"] == "500-1000"]
            ),
            "above_1000": self.process_subset(
                narrow_body_df[narrow_body_df["flown_distance_range"] == ">1000"]
            ),
            "non_b77w": self.process_subset(
                wide_body_df[wide_body_df["flown_distance_range"] == "NonB77W"]
            ),
            "b77w": self.process_subset(
                wide_body_df[wide_body_df["flown_distance_range"] == "B77W"]
            ),
        }

        return result
