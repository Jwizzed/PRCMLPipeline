from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image='gcr.io/prc-data-pipeline/ml-image',
    packages_to_install=["pandas", "fsspec", "gcsfs"])
def add_external_data(
        train_file: InputPath("CSV"),
        test_file: InputPath("CSV"),
        external_info_file: str,
        train_enriched_file: OutputPath("CSV"),
        test_enriched_file: OutputPath("CSV"),
):
    """Adds external aircraft information."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.enrich_data import DataEnricher

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    enricher = DataEnricher(external_info_file)
    train_enriched, test_enriched = enricher.enrich_data(train_df, test_df)

    train_enriched.to_csv(train_enriched_file, index=False)
    test_enriched.to_csv(test_enriched_file, index=False)
