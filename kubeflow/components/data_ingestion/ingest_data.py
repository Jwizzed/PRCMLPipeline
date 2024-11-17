from kfp.v2.dsl import component, OutputPath


@component(
    base_image="gcr.io/prc-data-pipeline/ml-image",
    packages_to_install=["pandas", "fsspec", "gcsfs", "pyarrow"],
)
def load_data(
    data_path: str,
    train_file: OutputPath("CSV"),
    test_file: OutputPath("CSV"),
    trajectory_file: OutputPath("CSV"),
):
    """Loads the flight data and trajectory data."""
    from kubeflow.src.data_ingestion.ingest_data import DataIngestor

    loader = DataIngestor(data_path)
    train_df, test_df, trajectory_df = loader.load_and_process_data()

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    trajectory_df.to_csv(trajectory_file, index=False)
