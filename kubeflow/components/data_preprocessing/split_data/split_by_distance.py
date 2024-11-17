from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image="gcr.io/prc-data-pipeline/ml-image",
    packages_to_install=["pandas", "scikit-learn"],
)
def split_by_distance(
    input_file: InputPath("CSV"),
    X_0_500_output: OutputPath("CSV"),
    X_500_1000_output: OutputPath("CSV"),
    X_above_1000_output: OutputPath("CSV"),
    X_non_b77w_output: OutputPath("CSV"),
    X_b77w_output: OutputPath("CSV"),
    y_0_500_output: OutputPath("CSV"),
    y_500_1000_output: OutputPath("CSV"),
    y_above_1000_output: OutputPath("CSV"),
    y_non_b77w_output: OutputPath("CSV"),
    y_b77w_output: OutputPath("CSV"),
):
    """Splits the data by flown distance."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.split_data.distance_split_stretegy import (
        DistanceSplitStrategy,
    )

    df = pd.read_csv(input_file)
    strategy = DistanceSplitStrategy()
    split_results = strategy.split(df)

    split_results["0-500"][0].to_csv(X_0_500_output, index=False)
    split_results["500-1000"][0].to_csv(X_500_1000_output, index=False)
    split_results["above_1000"][0].to_csv(X_above_1000_output, index=False)
    split_results["non_b77w"][0].to_csv(X_non_b77w_output, index=False)
    split_results["b77w"][0].to_csv(X_b77w_output, index=False)

    split_results["0-500"][1].to_csv(y_0_500_output, index=False)
    split_results["500-1000"][1].to_csv(y_500_1000_output, index=False)
    split_results["above_1000"][1].to_csv(y_above_1000_output, index=False)
    split_results["non_b77w"][1].to_csv(y_non_b77w_output, index=False)
    split_results["b77w"][1].to_csv(y_b77w_output, index=False)
