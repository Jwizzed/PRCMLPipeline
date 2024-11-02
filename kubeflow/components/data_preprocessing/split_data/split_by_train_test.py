from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas", "scikit-learn"])
def split_train_test(
        input_file: InputPath("CSV"),
        X_train_output: OutputPath("CSV"),
        X_test_output: OutputPath("CSV"),
        y_train_output: OutputPath("CSV"),
        y_test_output: OutputPath("CSV")
):
    """Splits the data into train and test sets."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_file)

    X = df.drop(['tow'], axis=1)
    y = df[['tow']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(X_train_output, index=False)
    X_test.to_csv(X_test_output, index=False)
    y_train.to_csv(y_train_output, index=False)
    y_test.to_csv(y_test_output, index=False)