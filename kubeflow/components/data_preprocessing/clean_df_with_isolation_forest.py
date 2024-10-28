from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    packages_to_install=['scikit-learn', 'pandas', 'numpy']
)
def clean_dataframe_with_isolation_forest(
        input_df: InputPath('CSV'),
        output_df: OutputPath('CSV'),
        contamination: float = 0.01
):
    """Cleans a dataframe using Isolation Forest for outlier detection."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest

    df = pd.read_csv(input_df)
    print("Shape before clean: ", df.shape)

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna()

    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_df[numeric_columns])

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(scaled_data)

    cleaned_df = cleaned_df[outlier_labels == 1].reset_index(drop=True)

    print("Shape after clean: ", cleaned_df.shape)
    cleaned_df.to_csv(output_df, index=False)


@component(
    packages_to_install=['scikit-learn', 'pandas', 'numpy']
)
def clean_trajectory_with_isolation_forest(
        input_df: InputPath('CSV'),
        output_df: OutputPath('CSV'),
        contamination: float = 0.01
):
    """Cleans a trajectory dataframe using Isolation Forest for outlier detection."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest

    df = pd.read_csv(input_df)
    print("Shape before clean: ", df.shape)

    cleaned_df = df.copy()

    cleaned_df = cleaned_df[
        (cleaned_df['altitude'] >= 0) & (cleaned_df['altitude'] <= 47000)]
    cleaned_df = cleaned_df.dropna()

    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns

    def clean_group(group):
        if len(group) <= 1:
            return group

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(group[numeric_columns])

        iso_forest = IsolationForest(contamination=contamination,
                                     random_state=42)
        outlier_labels = iso_forest.fit_predict(scaled_data)

        return group[outlier_labels == 1]

    cleaned_df = cleaned_df.groupby('flight_id').apply(clean_group)

    cleaned_df = cleaned_df.reset_index(drop=True)

    print("Shape after clean: ", cleaned_df.shape)
    cleaned_df.to_csv(output_df, index=False)
