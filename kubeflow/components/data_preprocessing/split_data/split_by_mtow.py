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
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(input_file)

    def process_subset(subset_df):
        subset_df = subset_df.copy()
        scaler = StandardScaler()
        subset_df['MTOW(kg)'] = scaler.fit_transform(subset_df[['MTOW(kg)']])

        if not is_test:
            X = subset_df.drop(['tow', 'MTOW_range'], axis=1).reset_index(
                drop=True)
            y = subset_df[['flight_id', 'tow']].drop('flight_id',
                                                     axis=1).reset_index(
                drop=True)
            return X, y
        else:
            X = subset_df.drop(['MTOW_range'], axis=1).reset_index(drop=True)
            return X, None

    narrow_body_df = df[df["MTOW(kg)"] <= 115680].copy()
    wide_body_df = df[df["MTOW(kg)"] > 115680].copy()

    narrow_body_df['MTOW_range'] = pd.qcut(narrow_body_df['MTOW(kg)'],
                                           q=4,
                                           labels=['Very Low', 'Low', 'Medium',
                                                   'High'])

    wide_body_df['MTOW_range'] = wide_body_df['aircraft_type'].apply(
        lambda x: 'B77W' if x == 'B77W' else 'NonB77W'
    )

    X_very_low, y_very_low = process_subset(
        narrow_body_df[narrow_body_df['MTOW_range'] == 'Very Low'])
    X_low, y_low = process_subset(
        narrow_body_df[narrow_body_df['MTOW_range'] == 'Low'])
    X_medium, y_medium = process_subset(
        narrow_body_df[narrow_body_df['MTOW_range'] == 'Medium'])
    X_high, y_high = process_subset(
        narrow_body_df[narrow_body_df['MTOW_range'] == 'High'])
    X_non_b77w, y_non_b77w = process_subset(
        wide_body_df[wide_body_df['MTOW_range'] == 'NonB77W'])
    X_b77w, y_b77w = process_subset(
        wide_body_df[wide_body_df['MTOW_range'] == 'B77W'])

    X_very_low.to_csv(X_very_low_output, index=False)
    X_low.to_csv(X_low_output, index=False)
    X_medium.to_csv(X_medium_output, index=False)
    X_high.to_csv(X_high_output, index=False)
    X_non_b77w.to_csv(X_non_b77w_output, index=False)
    X_b77w.to_csv(X_b77w_output, index=False)

    if not is_test:
        y_very_low.to_csv(y_very_low_output, index=False)
        y_low.to_csv(y_low_output, index=False)
        y_medium.to_csv(y_medium_output, index=False)
        y_high.to_csv(y_high_output, index=False)
        y_non_b77w.to_csv(y_non_b77w_output, index=False)
        y_b77w.to_csv(y_b77w_output, index=False)