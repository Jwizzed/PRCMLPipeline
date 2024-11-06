from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["scikit-learn", "pandas", "numpy"])
def clean_dataframe_with_isolation_forest(
        input_df: InputPath("CSV"),
        output_df: OutputPath("CSV"),
        contamination: float = 0.01,
):
    """Cleans a dataframe using Isolation Forest for outlier detection."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.clean_data import DataCleaner

    df = pd.read_csv(input_df)
    cleaned_df = DataCleaner.clean_dataframe(df, contamination)
    cleaned_df.to_csv(output_df, index=False)


@component(packages_to_install=["scikit-learn", "pandas", "numpy"])
def clean_trajectory_with_isolation_forest(
        input_df: InputPath("CSV"),
        output_df: OutputPath("CSV"),
        contamination: float = 0.01,
):
    """Cleans a trajectory dataframe using Isolation Forest for outlier detection."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.clean_data import DataCleaner

    df = pd.read_csv(input_df)
    cleaned_df = DataCleaner.clean_trajectory(df, contamination)
    cleaned_df.to_csv(output_df, index=False)
