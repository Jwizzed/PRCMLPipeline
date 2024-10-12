from typing import Tuple

import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning


@step
def clean_data(
    trajectory_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cleans the trajectory, train, and test data."""
    data_cleaning = DataCleaning()

    train_df_cleaned = data_cleaning.clean_train_test(train_df)
    test_df_cleaned = data_cleaning.clean_train_test(test_df)

    trajectory_df_cleaned = data_cleaning.clean_trajectory(trajectory_df)

    return trajectory_df_cleaned, train_df_cleaned, test_df_cleaned
