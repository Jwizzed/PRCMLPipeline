import pandas as pd


class DurationCalculator:
    @staticmethod
    def get_duration(df):
        df["actual_offblock_time"] = pd.to_datetime(df["actual_offblock_time"])
        df["arrival_time"] = pd.to_datetime(df["arrival_time"])
        df["duration"] = (
                                 df["arrival_time"] - df[
                             "actual_offblock_time"]
                         ).dt.total_seconds() / 60
        return df

    def calculate_duration(self, input_df):
        df = pd.read_csv(input_df)
        return self.get_duration(df)
