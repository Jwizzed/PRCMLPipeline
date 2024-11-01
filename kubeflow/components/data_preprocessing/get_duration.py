import pandas as pd
from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas"])
def calculate_flight_duration(input_df: InputPath("CSV"), output_df: OutputPath("CSV")):
    """Calculates flight duration in minutes."""

    def get_duration(df):
        df["actual_offblock_time"] = pd.to_datetime(df["actual_offblock_time"])
        df["arrival_time"] = pd.to_datetime(df["arrival_time"])
        df["duration"] = (
            df["arrival_time"] - df["actual_offblock_time"]
        ).dt.total_seconds() / 60
        return df

    df = pd.read_csv(input_df)
    df_with_duration = get_duration(df)
    df_with_duration.to_csv(output_df, index=False)
