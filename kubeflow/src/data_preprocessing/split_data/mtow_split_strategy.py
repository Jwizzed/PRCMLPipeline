from typing import Dict, Tuple, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler

from kubeflow.src.data_preprocessing.split_data.split_data_strategy import (
    SplitDataStrategy,
)


class MTOWSplitStrategy(SplitDataStrategy):
    def process_subset(
        self, subset_df: pd.DataFrame, is_test: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, None]]:
        subset_df = subset_df.copy()
        scaler = StandardScaler()
        subset_df["MTOW(kg)"] = scaler.fit_transform(subset_df[["MTOW(kg)"]])

        if not is_test:
            X = subset_df.drop(["tow", "MTOW_range"], axis=1).reset_index(drop=True)
            y = (
                subset_df[["flight_id", "tow"]]
                .drop("flight_id", axis=1)
                .reset_index(drop=True)
            )
            return X, y
        else:
            X = subset_df.drop(["MTOW_range"], axis=1).reset_index(drop=True)
            return X, None

    def split(
        self, df: pd.DataFrame, **kwargs
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        is_test = kwargs.get("is_test", False)

        narrow_body_df = df[df["MTOW(kg)"] <= 115680].copy()
        wide_body_df = df[df["MTOW(kg)"] > 115680].copy()

        narrow_body_df["MTOW_range"] = pd.qcut(
            narrow_body_df["MTOW(kg)"],
            q=4,
            labels=["Very Low", "Low", "Medium", "High"],
        )

        wide_body_df["MTOW_range"] = wide_body_df["aircraft_type"].apply(
            lambda x: "B77W" if x == "B77W" else "NonB77W"
        )

        result = {
            "very_low": self.process_subset(
                narrow_body_df[narrow_body_df["MTOW_range"] == "Very Low"], is_test
            ),
            "low": self.process_subset(
                narrow_body_df[narrow_body_df["MTOW_range"] == "Low"], is_test
            ),
            "medium": self.process_subset(
                narrow_body_df[narrow_body_df["MTOW_range"] == "Medium"], is_test
            ),
            "high": self.process_subset(
                narrow_body_df[narrow_body_df["MTOW_range"] == "High"], is_test
            ),
            "non_b77w": self.process_subset(
                wide_body_df[wide_body_df["MTOW_range"] == "NonB77W"], is_test
            ),
            "b77w": self.process_subset(
                wide_body_df[wide_body_df["MTOW_range"] == "B77W"], is_test
            ),
        }

        return result
