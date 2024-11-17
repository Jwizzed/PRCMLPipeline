from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image="gcr.io/prc-data-pipeline/ml-image",
    packages_to_install=["pandas", "scikit-learn"],
)
def split_train_test(
    input_file: InputPath("CSV"),
    X_train_output: OutputPath("CSV"),
    X_test_output: OutputPath("CSV"),
    y_train_output: OutputPath("CSV"),
    y_test_output: OutputPath("CSV"),
):
    """Splits the data into train and test sets."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.split_data.train_test_split_strategy import (
        TrainTestSplitStrategy,
    )

    df = pd.read_csv(input_file)
    strategy = TrainTestSplitStrategy()
    split_results = strategy.split(df)

    split_results["train"][0].to_csv(X_train_output, index=False)
    split_results["test"][0].to_csv(X_test_output, index=False)
    split_results["train"][1].to_csv(y_train_output, index=False)
    split_results["test"][1].to_csv(y_test_output, index=False)
