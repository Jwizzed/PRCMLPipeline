from typing import Tuple

import pandas as pd
from src.feature_engineering import FeatureEngineering
from zenml import step


@step
def feature_engineering(
        trajectory_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Performs feature engineering and prepares data for modeling."""

    fe = FeatureEngineering()
    external_information = fe.get_external_information()
    fe.external_df = pd.DataFrame.from_dict(external_information,
                                            orient='index')
    fe.external_df.reset_index(inplace=True)
    fe.external_df.rename(columns={'index': 'aircraft_type'}, inplace=True)

    fe.train_df = pd.merge(train_df, fe.external_df, on='aircraft_type',
                           how='left')
    fe.test_df = pd.merge(test_df, fe.external_df, on='aircraft_type',
                          how='left')

    aircraft_types_with_info = external_information.keys()
    fe.train_df = fe.train_df[
        fe.train_df['aircraft_type'].isin(aircraft_types_with_info)]
    fe.train_df.reset_index(drop=True, inplace=True)
    fe.test_df.reset_index(drop=True, inplace=True)

    td_df = fe.calculate_thrust_minus_drag_for_phases(trajectory_df)

    aggregated_features = fe.aggregate_features(td_df)

    merged_df = pd.merge(fe.train_df, aggregated_features, left_on='flight_id', right_on='flight_id', how='inner')

    exclude_cols = ['flight_id', 'tow', 'date', 'callsign', 'adep', 'ades',
                    'actual_offblock_time', 'arrival_time', 'aircraft_type',
                    'wtc', 'airline']

    merged_df = fe.normalize_dataframe(merged_df, exclude_columns=exclude_cols)
    merged_df = fe.encode_categorical_features(merged_df)
    merged_df = fe.drop_unnecessary_features(merged_df)

    X = merged_df.drop('tow', axis=1)
    y = merged_df['tow']

    return X, y
