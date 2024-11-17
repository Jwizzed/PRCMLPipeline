import pandas as pd


class DataIngestor:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_and_process_data(self):
        train_df = pd.read_csv(
            f"{self.data_path}/challenge_set.csv",
            parse_dates=["date", "actual_offblock_time", "arrival_time"],
        )

        test_df = pd.read_csv(
            f"{self.data_path}/submission_set.csv",
            parse_dates=["date", "actual_offblock_time", "arrival_time"],
        ).drop(["tow"], axis=1)

        # Format datetime columns
        datetime_columns = ["date", "actual_offblock_time", "arrival_time"]
        for col in datetime_columns:
            train_df[col] = train_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            test_df[col] = test_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        trajectory_df = pd.read_parquet(f"{self.data_path}/2022-01-01.parquet")

        return train_df, test_df, trajectory_df
