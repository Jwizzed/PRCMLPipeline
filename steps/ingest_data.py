from typing import Tuple, Union

import pandas as pd
from typing_extensions import Annotated
from zenml import step
from config import Config


@step
def ingest_data(config: Config) -> Union[
    Tuple[
        Annotated[pd.DataFrame, "Train df"],
        Annotated[pd.DataFrame, "Test df"],
    ],
    Tuple[
        Annotated[pd.DataFrame, "Trajectory df"],
        Annotated[pd.DataFrame, "Train df"],
        Annotated[pd.DataFrame, "Test df"],
    ]
]:
    """Reads the trajectory, train, and test data based on configuration."""
    challenge_file_path = "data/challenge_set.csv"
    submission_file_path = "data/submission_set.csv"

    train_df = pd.read_csv(
        challenge_file_path,
        parse_dates=["date", "actual_offblock_time", "arrival_time"],
    )

    test_df = pd.read_csv(
        submission_file_path,
        parse_dates=["date", "actual_offblock_time", "arrival_time"],
    ).drop(["tow"], axis=1)

    if config.USE_TRAJECTORY:
        trajectory_file_path = "data/2022-01-01.parquet"
        trajectory_df = pd.read_parquet(trajectory_file_path)
        return trajectory_df, train_df, test_df
    else:
        return train_df, test_df
