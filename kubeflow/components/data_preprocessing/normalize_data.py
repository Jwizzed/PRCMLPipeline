from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas", "scikit-learn"])
def normalize_dataframe(
    input_file: InputPath("CSV"),
    output_file: OutputPath("CSV"),
    exclude_columns: list,
    split_by_flown_distance: bool = False,
):
    """Normalizes the dataframe and excludes specified columns from normalization."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = pd.read_csv(input_file)

    if split_by_flown_distance:
        exclude_columns.append("flown_distance")

    df_normalized = df.copy()
    columns_to_normalize = df.select_dtypes(include=[np.number]).columns.difference(
        exclude_columns
    )

    scaler = StandardScaler()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    df_normalized.to_csv(output_file, index=False)
