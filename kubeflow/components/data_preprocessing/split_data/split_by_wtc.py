from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image="gcr.io/prc-data-pipeline/ml-image", packages_to_install=["pandas"]
)
def split_by_wtc(
    input_file: InputPath("CSV"),
    X_wtc_M_output: OutputPath("CSV"),
    X_wtc_H_output: OutputPath("CSV"),
    y_wtc_M_output: OutputPath("CSV"),
    y_wtc_H_output: OutputPath("CSV"),
):
    """Splits the data by Wake Turbulence Category (WTC)."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.split_data.wtc_split_strategy import (
        WTCSplitStrategy,
    )

    df = pd.read_csv(input_file)
    strategy = WTCSplitStrategy()
    split_results = strategy.split(df)

    split_results["wtc_M"][0].to_csv(X_wtc_M_output, index=False)
    split_results["wtc_H"][0].to_csv(X_wtc_H_output, index=False)
    split_results["wtc_M"][1].to_csv(y_wtc_M_output, index=False)
    split_results["wtc_H"][1].to_csv(y_wtc_H_output, index=False)
