from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas", "scikit-learn"])
def split_by_mtow(
        input_file: InputPath("CSV"),
        X_very_low_output: OutputPath("CSV"),
        X_low_output: OutputPath("CSV"),
        X_medium_output: OutputPath("CSV"),
        X_high_output: OutputPath("CSV"),
        X_non_b77w_output: OutputPath("CSV"),
        X_b77w_output: OutputPath("CSV"),
        y_very_low_output: OutputPath("CSV"),
        y_low_output: OutputPath("CSV"),
        y_medium_output: OutputPath("CSV"),
        y_high_output: OutputPath("CSV"),
        y_non_b77w_output: OutputPath("CSV"),
        y_b77w_output: OutputPath("CSV"),
        is_test: bool = False
):
    """Splits the data by Maximum Take-Off Weight (MTOW)."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.split_data.mtow_split_strategy import MTOWSplitStrategy

    df = pd.read_csv(input_file)
    strategy = MTOWSplitStrategy()
    split_results = strategy.split(df, is_test=is_test)

    # Index 0 for X and 1 for y
    split_results['very_low'][0].to_csv(X_very_low_output, index=False)
    split_results['low'][0].to_csv(X_low_output, index=False)
    split_results['medium'][0].to_csv(X_medium_output, index=False)
    split_results['high'][0].to_csv(X_high_output, index=False)
    split_results['non_b77w'][0].to_csv(X_non_b77w_output, index=False)
    split_results['b77w'][0].to_csv(X_b77w_output, index=False)

    if not is_test:
        split_results['very_low'][1].to_csv(y_very_low_output, index=False)
        split_results['low'][1].to_csv(y_low_output, index=False)
        split_results['medium'][1].to_csv(y_medium_output, index=False)
        split_results['high'][1].to_csv(y_high_output, index=False)
        split_results['non_b77w'][1].to_csv(y_non_b77w_output, index=False)
        split_results['b77w'][1].to_csv(y_b77w_output, index=False)
