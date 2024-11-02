from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas"])
def split_by_wtc(
        input_file: InputPath("CSV"),
        X_wtc_M_output: OutputPath("CSV"),
        X_wtc_H_output: OutputPath("CSV"),
        y_wtc_M_output: OutputPath("CSV"),
        y_wtc_H_output: OutputPath("CSV")
):
    """Splits the data by Wake Turbulence Category (WTC)."""
    import pandas as pd

    df = pd.read_csv(input_file)

    X = df.drop(['tow'], axis=1)
    y = df[['flight_id', 'tow']]

    X_wtc_M = X[X['wtc_M'] == 1].drop(["wtc_H", "wtc_M"], axis=1).reset_index(
        drop=True)
    X_wtc_H = X[X['wtc_H'] == 1].drop(["wtc_H", "wtc_M"], axis=1).reset_index(
        drop=True)

    y_wtc_M = y[y['flight_id'].isin(X_wtc_M.flight_id.unique())].drop(
        'flight_id', axis=1).reset_index(drop=True)
    y_wtc_H = y[y['flight_id'].isin(X_wtc_H.flight_id.unique())].drop(
        'flight_id', axis=1).reset_index(drop=True)

    X_wtc_M = X_wtc_M.drop('flight_id', axis=1).reset_index(drop=True)
    X_wtc_H = X_wtc_H.drop('flight_id', axis=1).reset_index(drop=True)

    X_wtc_M.to_csv(X_wtc_M_output, index=False)
    X_wtc_H.to_csv(X_wtc_H_output, index=False)
    y_wtc_M.to_csv(y_wtc_M_output, index=False)
    y_wtc_H.to_csv(y_wtc_H_output, index=False)