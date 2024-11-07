from typing import Dict, Tuple
import pandas as pd

from kubeflow.src.data_preprocessing.split_data.split_data_strategy import SplitDataStrategy


class WTCSplitStrategy(SplitDataStrategy):
    def process_subset(self, subset_df: pd.DataFrame, is_test: bool = False):
        pass  # Not needed for this strategy

    def split(self, df: pd.DataFrame, **kwargs) -> Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame]]:
        X = df.drop(['tow'], axis=1)
        y = df[['flight_id', 'tow']]

        X_wtc_M = X[X['wtc_M'] == 1].drop(["wtc_H", "wtc_M"],
                                          axis=1).reset_index(drop=True)
        X_wtc_H = X[X['wtc_H'] == 1].drop(["wtc_H", "wtc_M"],
                                          axis=1).reset_index(drop=True)

        y_wtc_M = y[y['flight_id'].isin(X_wtc_M.flight_id.unique())].drop(
            'flight_id', axis=1).reset_index(drop=True)
        y_wtc_H = y[y['flight_id'].isin(X_wtc_H.flight_id.unique())].drop(
            'flight_id', axis=1).reset_index(drop=True)

        X_wtc_M = X_wtc_M.drop('flight_id', axis=1).reset_index(drop=True)
        X_wtc_H = X_wtc_H.drop('flight_id', axis=1).reset_index(drop=True)

        return {
            'wtc_M': (X_wtc_M, y_wtc_M),
            'wtc_H': (X_wtc_H, y_wtc_H)
        }

