from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step

from src.data_cleaning import DataCleaning


@step
def clean_data(
    trajectory_df: Annotated[pd.DataFrame, "Trajectory df"],
    train_df: Annotated[pd.DataFrame, "Train df"],
    test_df: Annotated[pd.DataFrame, "Test df"],
) -> Tuple[
    Annotated[pd.DataFrame, "Cleaned trajectory df"],
    Annotated[pd.DataFrame, "Cleaned train df"],
    Annotated[pd.DataFrame, "Cleaned test df"],
]:
    """Cleans the trajectory, train, and test data."""
    data_cleaning = DataCleaning()

    train_df_cleaned = data_cleaning.clean_train_test(train_df)
    test_df_cleaned = data_cleaning.clean_train_test(test_df)

    trajectory_df_cleaned = data_cleaning.clean_trajectory(trajectory_df)

    return trajectory_df_cleaned, train_df_cleaned, test_df_cleaned
