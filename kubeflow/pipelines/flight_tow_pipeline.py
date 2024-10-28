from kfp.v2 import dsl
from kfp.v2.dsl import component, InputPath, OutputPath, Metrics, Output


@component(
    packages_to_install=['pandas', 'scikit-learn']
)
def clean_with_isolation_forest(
        train_file: InputPath('CSV'),
        test_file: InputPath('CSV'),
        train_cleaned_file: OutputPath('CSV'),
        test_cleaned_file: OutputPath('CSV')
):
    """Cleans data using Isolation Forest."""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    def clean_dataframe_with_isolation_forest(df, contamination=0.01):
        cleaned_df = df.copy()
        cleaned_df = cleaned_df.dropna()

        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cleaned_df[numeric_columns])

        iso_forest = IsolationForest(contamination=contamination,
                                     random_state=42)
        outlier_labels = iso_forest.fit_predict(scaled_data)
        cleaned_df = cleaned_df[outlier_labels == 1].reset_index(drop=True)

        return cleaned_df

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_cleaned = clean_dataframe_with_isolation_forest(train_df)
    test_cleaned = clean_dataframe_with_isolation_forest(test_df)

    train_cleaned.to_csv(train_cleaned_file, index=False)
    test_cleaned.to_csv(test_cleaned_file, index=False)


@component(
    packages_to_install=['pandas', 'scikit-learn']
)
def encode_features(
        train_file: InputPath('CSV'),
        test_file: InputPath('CSV'),
        train_encoded_file: OutputPath('CSV'),
        test_encoded_file: OutputPath('CSV')
):
    """Encodes categorical features."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    def encode_categorical_features(df):
        categorical_col = ["adep", "country_code_adep", "ades",
                           "country_code_ades",
                           "aircraft_type", "airline"]

        encoder = LabelEncoder()
        for col in categorical_col:
            df[col + "_encoded"] = encoder.fit_transform(df[col])
            df = df.drop(columns=[col])

        df = pd.get_dummies(df, columns=["wtc"])
        df["wtc_M"] = df["wtc_M"].astype(int)
        df["wtc_H"] = df["wtc_H"].astype(int)

        return df

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_encoded = encode_categorical_features(train_df)
    test_encoded = encode_categorical_features(test_df)

    train_encoded.to_csv(train_encoded_file, index=False)
    test_encoded.to_csv(test_encoded_file, index=False)


@component(
    packages_to_install=['pandas']
)
def select_feature(
        train_file: InputPath('CSV'),
        test_file: InputPath('CSV'),
        train_selected_feature_file: OutputPath('CSV'),
        test_selected_feature_file: OutputPath('CSV'),
):
    import pandas as pd

    def drop_features(df):
        drop_cols = [
            # "flight_id",
            "date",
            "callsign",
            "name_adep",
            "name_ades",
            "actual_offblock_time",
            "arrival_time",
        ]
        dropped_df = df.drop(columns=drop_cols, errors='ignore')

        return dropped_df

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df = drop_features(train_df)
    test_df = drop_features(test_df)

    train_df.to_csv(train_selected_feature_file, index=False)
    test_df.to_csv(test_selected_feature_file, index=False)


@component(
    packages_to_install=['pandas', 'scikit-learn', 'xgboost', 'catboost']
)
def train_ensemble_model(
        train_file: InputPath('CSV'),
        test_file: InputPath('CSV'),
        catboost_model_file: OutputPath('Joblib'),
        xgboost_model_file: OutputPath('Joblib'),
        predictions_file: OutputPath('CSV'),
        metrics_file: OutputPath('CSV'),
        parameters_file: OutputPath('JSON'),
        metrics: Output[Metrics]
):
    """Trains an ensemble of CatBoost and XGBoost models."""
    import pandas as pd
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import joblib
    import json

    train_df = pd.read_csv(train_file)

    X_train = train_df.drop(['tow'], axis=1)
    y_train = train_df['tow']


    catboost_params = {
        'random_state': 42,
        'depth': 9,
        'iterations': 2000,
        'learning_rate': 0.15
    }

    xgboost_params = {
        'random_state': 42,
        'learning_rate': 0.05,
        'max_depth': 8,
        'n_estimators': 2000
    }

    parameters = {
        'train_samples': len(X_train),
        'features': list(X_train.columns),
        'catboost_params': catboost_params,
        'xgboost_params': xgboost_params
    }

    with open(parameters_file, 'w') as f:
        json.dump(parameters, f)

    catboost_model = CatBoostRegressor(**catboost_params)
    catboost_model.fit(X_train, y_train)
    catboost_pred = catboost_model.predict(X_train)
    catboost_rmse = np.sqrt(mean_squared_error(y_train, catboost_pred))

    xgboost_model = XGBRegressor(**xgboost_params)
    xgboost_model.fit(X_train, y_train)
    xgboost_pred = xgboost_model.predict(X_train)
    xgboost_rmse = np.sqrt(mean_squared_error(y_train, xgboost_pred))

    metrics.log_metric("catboost_rmse", catboost_rmse)
    metrics.log_metric("xgboost_rmse", xgboost_rmse)

    ensemble_pred = (catboost_pred + xgboost_pred) / 2
    ensemble_rmse = np.sqrt(mean_squared_error(y_train, ensemble_pred))
    metrics.log_metric("ensemble_rmse", ensemble_rmse)

    joblib.dump(catboost_model, catboost_model_file)
    joblib.dump(xgboost_model, xgboost_model_file)

    results_df = pd.DataFrame({
        'flight_id': X_train['flight_id'],
        'tow': ensemble_pred
    })
    results_df.to_csv(predictions_file, index=False)

    metrics_df = pd.DataFrame({
        'metric': [
            'CatBoost RMSE',
            'XGBoost RMSE',
            'Ensemble RMSE'
        ],
        'value': [
            catboost_rmse,
            xgboost_rmse,
            ensemble_rmse
        ]
    })
    metrics_df.to_csv(metrics_file, index=False)


@dsl.pipeline(
    name='Flight TOW Prediction Pipeline',
    description='Pipeline for predicting Take-Off Weight (TOW)'
)
def flight_tow_pipeline(data_path: str):
    load_data_task = load_data(data_path=data_path)

    external_data_task = add_external_data(
        train_file=load_data_task.outputs['train_file'],
        test_file=load_data_task.outputs['test_file']
    )

    clean_data_task = clean_with_isolation_forest(
        train_file=external_data_task.outputs['train_enriched_file'],
        test_file=external_data_task.outputs['test_enriched_file']
    )

    encode_task = encode_features(
        train_file=clean_data_task.outputs['train_cleaned_file'],
        test_file=clean_data_task.outputs['test_cleaned_file']
    )

    select_feature_task = select_feature(
        train_file=encode_task.outputs['train_encoded_file'],
        test_file=encode_task.outputs['test_encoded_file']
    )

    train_model_task = train_ensemble_model(
        train_file=select_feature_task.outputs['train_selected_feature_file'],
        test_file=select_feature_task.outputs['test_selected_feature_file']
    )
