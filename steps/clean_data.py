from typing import Tuple, Union

import pandas as pd
from typing_extensions import Annotated
from zenml import step

from src.data_cleaning import DataCleaning
from config import Config


@step
def clean_data(
        config: Config,
        data: Union[
            Tuple[
                Annotated[pd.DataFrame, "Train df"],
                Annotated[pd.DataFrame, "Test df"]],
            Tuple[
                Annotated[pd.DataFrame, "Trajectory df"],
                Annotated[pd.DataFrame, "Train df"],
                Annotated[pd.DataFrame, "Test df"],
            ],
        ],
) -> Union[
    Tuple[
        Annotated[pd.DataFrame, "Cleaned train df"],
        Annotated[pd.DataFrame, "Cleaned test df"],
    ],
    Tuple[
        Annotated[pd.DataFrame, "Cleaned trajectory df"],
        Annotated[pd.DataFrame, "Cleaned train df"],
        Annotated[pd.DataFrame, "Cleaned test df"],
    ],
]:
    """Cleans the trajectory, train, and test data."""
    data_cleaning = DataCleaning()

    if config.USE_TRAJECTORY:
        trajectory_df, train_df, test_df = data
        trajectory_df_cleaned = data_cleaning.clean_trajectory(trajectory_df)
    else:
        train_df, test_df = data
        trajectory_df_cleaned = None

    train_df_cleaned = data_cleaning.clean_train_test(train_df)
    test_df_cleaned = data_cleaning.clean_train_test(test_df)

    if config.USE_TRAJECTORY:
        return trajectory_df_cleaned, train_df_cleaned, test_df_cleaned
    else:
        return train_df_cleaned, test_df_cleaned
