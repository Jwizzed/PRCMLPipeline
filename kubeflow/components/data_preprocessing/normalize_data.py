from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image="gcr.io/prc-data-pipeline/ml-image",
    packages_to_install=["pandas", "scikit-learn"],
)
def normalize_dataframe(
    input_file: InputPath("CSV"),
    output_file: OutputPath("CSV"),
    exclude_columns: list,
    split_by_flown_distance: bool = False,
):
    """Normalizes the dataframe and excludes specified columns from normalization."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.normalize_data import DataNormalizer

    df = pd.read_csv(input_file)

    normalizer = DataNormalizer(exclude_columns, split_by_flown_distance)
    df_normalized = normalizer.normalize_data(df)

    df_normalized.to_csv(output_file, index=False)
