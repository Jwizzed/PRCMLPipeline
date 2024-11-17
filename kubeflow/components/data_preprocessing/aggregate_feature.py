from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image="gcr.io/prc-data-pipeline/ml-image",
    packages_to_install=["pandas", "numpy", "tqdm", "scipy"],
)
def calculate_and_aggregate_features(
    trajectory_df_path: InputPath("CSV"),
    train_df_path: InputPath("CSV"),
    aggregated_features_path: OutputPath("CSV"),
    use_trajectory: bool,
    flight_phases_refinement: bool,
):
    """Calculates thrust minus drag for flight phases and aggregates features."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.feature_engineering import FeatureEngineering

    trajectory_df = pd.read_csv(trajectory_df_path)
    train_df = pd.read_csv(train_df_path)

    feature_engineering = FeatureEngineering()
    merged_df = feature_engineering.process_data(
        trajectory_df, train_df, use_trajectory, flight_phases_refinement
    )

    merged_df.to_csv(aggregated_features_path, index=False)
