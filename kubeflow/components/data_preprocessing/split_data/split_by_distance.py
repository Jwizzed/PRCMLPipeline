from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas", "scikit-learn"])
def split_by_distance(
        input_file: InputPath("CSV"),
        X_0_500_output: OutputPath("CSV"),
        X_500_1000_output: OutputPath("CSV"),
        X_above_1000_output: OutputPath("CSV"),
        X_non_b77w_output: OutputPath("CSV"),
        X_b77w_output: OutputPath("CSV"),
        y_0_500_output: OutputPath("CSV"),
        y_500_1000_output: OutputPath("CSV"),
        y_above_1000_output: OutputPath("CSV"),
        y_non_b77w_output: OutputPath("CSV"),
        y_b77w_output: OutputPath("CSV")
):
    """Splits the data by flown distance."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(input_file)

    def process_subset(subset_df):
        subset_df = subset_df.copy()
        scaler = StandardScaler()
        subset_df[['MTOW(kg)', 'flown_distance']] = scaler.fit_transform(
            subset_df[['MTOW(kg)', 'flown_distance']]
        )

        X = subset_df.drop(['tow', 'flown_distance_range'],
                           axis=1).reset_index(drop=True)
        y = subset_df[['flight_id', 'tow']].drop('flight_id',
                                                 axis=1).reset_index(drop=True)
        return X, y

    narrow_body_df = df[df["MTOW(kg)"] <= 115680].copy()
    wide_body_df = df[df["MTOW(kg)"] > 115680].copy()

    bins = [0, 500, 1000, float('inf')]
    labels = ['0-500', '500-1000', '>1000']
    narrow_body_df['flown_distance_range'] = pd.cut(
        narrow_body_df['flown_distance'],
        bins=bins,
        labels=labels,
        right=False
    )

    wide_body_df['flown_distance_range'] = wide_body_df['aircraft_type'].apply(
        lambda x: 'B77W' if x == 'B77W' else 'NonB77W'
    )

    X_0_500, y_0_500 = process_subset(
        narrow_body_df[narrow_body_df['flown_distance_range'] == '0-500'])
    X_500_1000, y_500_1000 = process_subset(
        narrow_body_df[narrow_body_df['flown_distance_range'] == '500-1000'])
    X_above_1000, y_above_1000 = process_subset(
        narrow_body_df[narrow_body_df['flown_distance_range'] == '>1000'])
    X_non_b77w, y_non_b77w = process_subset(
        wide_body_df[wide_body_df['flown_distance_range'] == 'NonB77W'])
    X_b77w, y_b77w = process_subset(
        wide_body_df[wide_body_df['flown_distance_range'] == 'B77W'])

    X_0_500.to_csv(X_0_500_output, index=False)
    X_500_1000.to_csv(X_500_1000_output, index=False)
    X_above_1000.to_csv(X_above_1000_output, index=False)
    X_non_b77w.to_csv(X_non_b77w_output, index=False)
    X_b77w.to_csv(X_b77w_output, index=False)

    y_0_500.to_csv(y_0_500_output, index=False)
    y_500_1000.to_csv(y_500_1000_output, index=False)
    y_above_1000.to_csv(y_above_1000_output, index=False)
    y_non_b77w.to_csv(y_non_b77w_output, index=False)
    y_b77w.to_csv(y_b77w_output, index=False)