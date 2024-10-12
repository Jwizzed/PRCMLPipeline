from zenml import step
import pandas as pd
from typing import Tuple


@step
def ingest_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads the trajectory, train, and test data."""
    trajectory_file_path = 'data/2022-01-01.parquet'
    challenge_file_path = 'data/challenge_set.csv'
    submission_file_path = 'data/submission_set.csv'

    trajectory_df = pd.read_parquet(trajectory_file_path)
    trajectory_df = trajectory_df[:len(trajectory_df) // 10]  # For the experiment

    train_df = pd.read_csv(challenge_file_path,
                           parse_dates=['date', 'actual_offblock_time',
                                        'arrival_time'])

    test_df = pd.read_csv(submission_file_path,
                          parse_dates=['date', 'actual_offblock_time',
                                       'arrival_time']).drop(["tow"], axis=1)

    return trajectory_df, train_df, test_df
