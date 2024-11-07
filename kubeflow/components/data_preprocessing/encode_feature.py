from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image='gcr.io/prc-data-pipeline/ml-image',
    packages_to_install=["pandas", "scikit-learn"])
def encode_categorical_features(
        input_file: InputPath("CSV"), output_file: OutputPath("CSV"),
        preserve_columns: list
):
    """Encodes categorical features in the dataframe, with the option to preserve certain columns."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.encode_feature import FeatureEncoder

    df = pd.read_csv(input_file)
    encoder = FeatureEncoder(preserve_columns)
    df_encoded = encoder.encode_features(df)
    df_encoded.to_csv(output_file, index=False)
