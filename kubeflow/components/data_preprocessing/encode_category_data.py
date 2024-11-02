from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas", "scikit-learn"])
def encode_categorical_features(
    input_file: InputPath("CSV"), output_file: OutputPath("CSV"), preserve_columns: list
):
    """Encodes categorical features in the dataframe, with the option to preserve certain columns."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(input_file)

    df_encoded = df.copy()

    categorical_col = [
        "adep",
        "country_code_adep",
        "ades",
        "country_code_ades",
        "aircraft_type",
        "airline",
    ]

    encoder = LabelEncoder()

    for col in categorical_col:
        df_encoded[col + "_encoded"] = encoder.fit_transform(df_encoded[col])
        if col not in preserve_columns:
            df_encoded = df_encoded.drop(columns=[col])

    oneHot_col = ["wtc"]
    df_encoded = pd.get_dummies(df_encoded, columns=oneHot_col)

    df_encoded["wtc_M"] = df_encoded["wtc_M"].astype(int)
    df_encoded["wtc_H"] = df_encoded["wtc_H"].astype(int)

    df_encoded.to_csv(output_file, index=False)
