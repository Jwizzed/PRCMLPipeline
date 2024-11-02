from kfp.v2.dsl import component, OutputPath


@component(packages_to_install=["pandas", "fsspec", "gcsfs", "pyarrow"])
def load_data(
        data_path: str,
        train_file: OutputPath("CSV"),
        test_file: OutputPath("CSV"),
        trajectory_file: OutputPath("CSV")
):
    """Loads the flight data and trajectory data."""
    import pandas as pd

    # Load main flight data
    train_df = pd.read_csv(
        f"{data_path}/challenge_set.csv",
        parse_dates=["date", "actual_offblock_time", "arrival_time"]
    )

    test_df = pd.read_csv(
        f"{data_path}/submission_set.csv",
        parse_dates=["date", "actual_offblock_time", "arrival_time"]
    ).drop(["tow"], axis=1)

    # Format datetime columns
    datetime_columns = ["date", "actual_offblock_time", "arrival_time"]
    for col in datetime_columns:
        train_df[col] = train_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        test_df[col] = test_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    trajectory_df = pd.read_parquet(f"{data_path}/2022-01-01.parquet")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    trajectory_df.to_csv(trajectory_file, index=False)
