from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas"])
def drop_features(
    input_file: InputPath("CSV"),
    output_file: OutputPath("CSV"),
    final_drop: bool = False,
):
    """Drops specified features from the dataframe."""
    import pandas as pd

    df = pd.read_csv(input_file)

    drop_cols = [
        "date",
        "callsign",
        "name_adep",
        "name_ades",
        "actual_offblock_time",
        "arrival_time",
    ]

    if final_drop:
        drop_cols.extend(["aircraft_type"])

    dropped_df = df.drop(columns=drop_cols, errors="ignore")
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
    from sklearn.feature_selection import SelectKBest, f_regression

    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file)
    X_test = pd.read_csv(X_test_file)

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.get_support()]

    selected_features_df = pd.DataFrame(
        selected_features, columns=["Selected Features"]
    )

    feature_scores_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Score": selector.scores_,
            "p-value": selector.pvalues_,
        }
    )

    X_train_selected = pd.DataFrame(
        selector.transform(X_train), columns=selected_features, index=X_train.index
    )

    X_test_selected = pd.DataFrame(
        selector.transform(X_test), columns=selected_features, index=X_test.index
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
    from sklearn.model_selection import train_test_split

    X = pd.read_csv(X_file)
    y = pd.read_csv(y_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(X_train_output, index=False)
    X_test.to_csv(X_test_output, index=False)
    y_train.to_csv(y_train_output, index=False)
    y_test.to_csv(y_test_output, index=False)

