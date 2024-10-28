from kfp.v2.dsl import component, OutputPath


@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs']
)
def load_data(
        data_path: str,
        train_file: OutputPath('CSV'),
        test_file: OutputPath('CSV')
):
    """Loads and preprocesses the data."""
    import pandas as pd

    train_df = pd.read_csv(f"{data_path}/challenge_set.csv",
                           parse_dates=['date', 'actual_offblock_time',
                                        'arrival_time'])
    test_df = pd.read_csv(f"{data_path}/submission_set.csv",
                          parse_dates=['date', 'actual_offblock_time',
                                       'arrival_time']).drop(["tow"], axis=1)

    def get_duration(df):
        df['duration'] = (df['arrival_time'] - df[
            'actual_offblock_time']).dt.total_seconds() / 60
        return df

    train_df = get_duration(train_df)
    test_df = get_duration(test_df)

    datetime_columns = ['date', 'actual_offblock_time', 'arrival_time']
    for col in datetime_columns:
        train_df[col] = train_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        test_df[col] = test_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)