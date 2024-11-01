from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas"])
def drop_features_component(
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
def feature_selection_component(
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
def split_wtc_component(
    input_file: InputPath("CSV"),
    X_wtc_M_file: OutputPath("CSV"),
    X_wtc_H_file: OutputPath("CSV"),
    y_wtc_M_file: OutputPath("CSV"),
    y_wtc_H_file: OutputPath("CSV"),
):
    """Splits data by Wake Turbulence Category."""
    import pandas as pd

    df = pd.read_csv(input_file)

    X = df.drop(["tow"], axis=1)
    y = df[["flight_id", "tow"]]

    X_wtc_M = X[X["wtc_M"] == 1].drop(["wtc_H", "wtc_M"], axis=1).reset_index(drop=True)
    X_wtc_H = X[X["wtc_H"] == 1].drop(["wtc_H", "wtc_M"], axis=1).reset_index(drop=True)

    y_wtc_M = (
        y[y["flight_id"].isin(X_wtc_M.flight_id.unique())]
        .drop("flight_id", axis=1)
        .reset_index(drop=True)
    )
    y_wtc_H = (
        y[y["flight_id"].isin(X_wtc_H.flight_id.unique())]
        .drop("flight_id", axis=1)
        .reset_index(drop=True)
    )

    X_wtc_M = X_wtc_M.drop("flight_id", axis=1).reset_index(drop=True)
    X_wtc_H = X_wtc_H.drop("flight_id", axis=1).reset_index(drop=True)

    X_wtc_M.to_csv(X_wtc_M_file, index=False)
    X_wtc_H.to_csv(X_wtc_H_file, index=False)
    y_wtc_M.to_csv(y_wtc_M_file, index=False)
    y_wtc_H.to_csv(y_wtc_H_file, index=False)


@component(packages_to_install=["pandas", "scikit-learn"])
def train_test_split_component(
    input_file: InputPath("CSV"),
    X_train_file: OutputPath("CSV"),
    X_test_file: OutputPath("CSV"),
    y_train_file: OutputPath("CSV"),
    y_test_file: OutputPath("CSV"),
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Performs train-test split on the input data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_file)
    X = df.drop(["tow"], axis=1)
    y = df[["tow"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(X_train_file, index=False)
    X_test.to_csv(X_test_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    y_test.to_csv(y_test_file, index=False)
