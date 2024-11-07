from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from kubeflow.src.data_preprocessing.split_data.split_data_strategy import SplitDataStrategy


class TrainTestSplitStrategy(SplitDataStrategy):
    def process_subset(self, subset_df: pd.DataFrame, is_test: bool = False):
        pass

    def split(self, df: pd.DataFrame, **kwargs) -> Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame]]:
        X = df.drop(['tow'], axis=1)
        y = df[['tow']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }
