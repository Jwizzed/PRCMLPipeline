from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas"])
def drop_features(
        input_file: InputPath("CSV"),
        output_file: OutputPath("CSV"),
        final_drop: bool = False,
):
    """Drops specified features from the dataframe."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.process_feature import FeatureProcessor

    df = pd.read_csv(input_file)
    dropped_df = FeatureProcessor.drop_features(df, final_drop)
    dropped_df.to_csv(output_file, index=False)


@component(packages_to_install=["pandas", "scikit-learn"])
def feature_selection(
        X_train_file: InputPath("CSV"),
        y_train_file: InputPath("CSV"),
        X_test_file: InputPath("CSV"),
        X_train_selected_file: OutputPath("CSV"),
        X_test_selected_file: OutputPath("CSV"),
        selected_features_file: OutputPath("CSV"),
        feature_scores_file: OutputPath("CSV"),
        k: int = 15,
):
    """Performs feature selection using SelectKBest."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.process_feature import FeatureProcessor

    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file)
    X_test = pd.read_csv(X_test_file)

    (X_train_selected, X_test_selected,
     selected_features_df,
     feature_scores_df) = FeatureProcessor.select_features(
        X_train, y_train, X_test, k
    )

    X_train_selected.to_csv(X_train_selected_file, index=False)
    X_test_selected.to_csv(X_test_selected_file, index=False)
    selected_features_df.to_csv(selected_features_file, index=False)
    feature_scores_df.to_csv(feature_scores_file, index=False)


@component(packages_to_install=["pandas", "scikit-learn"])
def process_category_split(
        X_file: InputPath("CSV"),
        y_file: InputPath("CSV"),
        X_train_output: OutputPath("CSV"),
        X_test_output: OutputPath("CSV"),
        y_train_output: OutputPath("CSV"),
        y_test_output: OutputPath("CSV"),
):
    """Performs train-test split and prints shapes for a category."""
    import pandas as pd
    from kubeflow.src.data_preprocessing.process_feature import FeatureProcessor

    X = pd.read_csv(X_file)
    y = pd.read_csv(y_file)

    X_train, X_test, y_train, y_test = FeatureProcessor.split_data(X, y)

    X_train.to_csv(X_train_output, index=False)
    X_test.to_csv(X_test_output, index=False)
    y_train.to_csv(y_train_output, index=False)
    y_test.to_csv(y_test_output, index=False)
