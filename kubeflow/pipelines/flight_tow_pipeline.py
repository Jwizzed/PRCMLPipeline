from kfp.v2 import dsl
from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs']
)
def load_data(
        data_path: str,
        train_file: OutputPath('CSV'),
        test_file: OutputPath('CSV')
):
    """Loads and preprocesses the data."""
    import pandas as pd

    train_df = pd.read_csv(f"{data_path}/challenge_set.csv",
                           parse_dates=['date', 'actual_offblock_time',
                                        'arrival_time'])
    test_df = pd.read_csv(f"{data_path}/submission_set.csv",
                          parse_dates=['date', 'actual_offblock_time',
                                       'arrival_time']).drop(["tow"], axis=1)

    def get_duration(df):
        df['duration'] = (df['arrival_time'] - df[
            'actual_offblock_time']).dt.total_seconds() / 60
        return df

    train_df = get_duration(train_df)
    test_df = get_duration(test_df)

    datetime_columns = ['date', 'actual_offblock_time', 'arrival_time']
    for col in datetime_columns:
        train_df[col] = train_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        test_df[col] = test_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)


@component(
    packages_to_install=['pandas']
)
def add_external_data(
        train_file: InputPath('CSV'),
        test_file: InputPath('CSV'),
        train_enriched_file: OutputPath('CSV'),
        test_enriched_file: OutputPath('CSV')
):
    """Adds external aircraft information."""
    import pandas as pd

    external_information = {
        "B738": {
            "MTOW(kg)": 70530,
            "passengers": 162,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 145,
        },
        "A333": {
            "MTOW(kg)": 230000,
            "passengers": 295,
            "ROC_Initial_Climb(ft/min)": 2000,
            "V2 (IAS)": 145,
        },
        "B77W": {
            "MTOW(kg)": 351500,
            "passengers": 365,
            "ROC_Initial_Climb(ft/min)": 2000,
            "V2 (IAS)": 149,
        },
        "B38M": {
            "MTOW(kg)": 82600,
            "passengers": 162,
            "ROC_Initial_Climb(ft/min)": 2500,
            "V2 (IAS)": 145,
        },
        "A320": {
            "MTOW(kg)": 73900,
            "passengers": 150,
            "ROC_Initial_Climb(ft/min)": 2500,
            "V2 (IAS)": 145,
        },
        "E190": {
            "MTOW(kg)": 45995,
            "passengers": 94,
            "ROC_Initial_Climb(ft/min)": 3400,
            "V2 (IAS)": 138,
        },
        "CRJ9": {
            "MTOW(kg)": 38330,
            "passengers": 80,
            "ROC_Initial_Climb(ft/min)": 2500,
            "V2 (IAS)": 140,
        },
        "A21N": {
            "MTOW(kg)": 97000,
            "passengers": 180,
            "ROC_Initial_Climb(ft/min)": 2000,
            "V2 (IAS)": 145,
        },
        "A20N": {
            "MTOW(kg)": 79000,
            "passengers": 150,
            "ROC_Initial_Climb(ft/min)": 2200,
            "V2 (IAS)": 145,
        },
        "B739": {
            "MTOW(kg)": 79015,
            "passengers": 177,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 149,
        },
        "BCS3": {
            "MTOW(kg)": 69900,
            "passengers": 120,
            "ROC_Initial_Climb(ft/min)": 3100,
            "V2 (IAS)": 165,
        },
        "E195": {
            "MTOW(kg)": 52290,
            "passengers": 100,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 140,
        },
        "A321": {
            "MTOW(kg)": 83000,
            "passengers": 185,
            "ROC_Initial_Climb(ft/min)": 2500,
            "V2 (IAS)": 145,
        },
        "A359": {
            "MTOW(kg)": 268000,
            "passengers": 314,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 150,
        },
        "A319": {
            "MTOW(kg)": 64000,
            "passengers": 124,
            "ROC_Initial_Climb(ft/min)": 2500,
            "V2 (IAS)": 135,
        },
        "A332": {
            "MTOW(kg)": 230000,
            "passengers": 253,
            "ROC_Initial_Climb(ft/min)": 2000,
            "V2 (IAS)": 145,
        },
        "B788": {
            "MTOW(kg)": 228000,
            "passengers": 210,
            "ROC_Initial_Climb(ft/min)": 2700,
            "V2 (IAS)": 165,
        },
        "B789": {
            "MTOW(kg)": 253000,
            "passengers": 406,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 165,
        },
        "BCS1": {
            "MTOW(kg)": 63100,
            "passengers": 100,
            "ROC_Initial_Climb(ft/min)": 3500,
            "V2 (IAS)": 140,
        },
        "B763": {
            "MTOW(kg)": 186880,
            "passengers": 269,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 160,
        },
        "AT76": {
            "MTOW(kg)": 23000,
            "passengers": 78,
            "ROC_Initial_Climb(ft/min)": 1350,
            "V2 (IAS)": 116,
        },
        "B772": {
            "MTOW(kg)": 247210,
            "passengers": 305,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 170,
        },
        "B737": {
            "MTOW(kg)": 66320,
            "passengers": 128,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 150,
        },
        "A343": {
            "MTOW(kg)": 275000,
            "passengers": 295,
            "ROC_Initial_Climb(ft/min)": 1400,
            "V2 (IAS)": 145,
        },
        "B39M": {
            "MTOW(kg)": 88300,
            "passengers": 178,
            "ROC_Initial_Climb(ft/min)": 2300,
            "V2 (IAS)": 150,
        },
        "B752": {
            "MTOW(kg)": 115680,
            "passengers": 200,
            "ROC_Initial_Climb(ft/min)": 3500,
            "V2 (IAS)": 145,
        },
        "B773": {
            "MTOW(kg)": 299370,
            "passengers": 368,
            "ROC_Initial_Climb(ft/min)": 3000,
            "V2 (IAS)": 168,
        },
        "E290": {
            "MTOW(kg)": 45995,
            "passengers": 94,
            "ROC_Initial_Climb(ft/min)": 3400,
            "V2 (IAS)": 138,
        },
    }

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    external_df = pd.DataFrame.from_dict(external_information, orient='index')
    external_df.reset_index(inplace=True)
    external_df.rename(columns={'index': 'aircraft_type'}, inplace=True)

    train_enriched = pd.merge(train_df, external_df, on='aircraft_type',
                              how='left')
    test_enriched = pd.merge(test_df, external_df, on='aircraft_type',
                             how='left')

    train_enriched.to_csv(train_enriched_file, index=False)
    test_enriched.to_csv(test_enriched_file, index=False)


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
    packages_to_install=['pandas', 'scikit-learn', 'xgboost', 'catboost',
                         'mlflow']
)
def train_ensemble_model(
        train_file: InputPath('CSV'),
        test_file: InputPath('CSV'),
        catboost_model_file: OutputPath('Joblib'),
        xgboost_model_file: OutputPath('Joblib'),
        predictions_file: OutputPath('CSV'),
        metrics_file: OutputPath('CSV')
):
    """Trains ensemble of CatBoost and XGBoost models."""
    import pandas as pd
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    import joblib
    import mlflow

    mlflow.start_run()

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(['tow'], axis=1)
    y_train = train_df['tow']

    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("features", list(X_train.columns))

    catboost_params = {
        'random_state': 42,
        'depth': 9,
        'iterations': 2000,
        'learning_rate': 0.15
    }
    mlflow.log_params({"catboost_" + k: v for k, v in catboost_params.items()})

    catboost_model = CatBoostRegressor(**catboost_params)
    catboost_model.fit(X_train, y_train)

    catboost_cv_scores = cross_val_score(
        catboost_model, X_train, y_train,
        cv=5, scoring='neg_root_mean_squared_error'
    )

    xgboost_params = {
        'random_state': 42,
        'learning_rate': 0.05,
        'max_depth': 8,
        'n_estimators': 2000
    }
    mlflow.log_params({"xgboost_" + k: v for k, v in xgboost_params.items()})

    xgboost_model = XGBRegressor(**xgboost_params)
    xgboost_model.fit(X_train, y_train)

    xgboost_cv_scores = cross_val_score(
        xgboost_model, X_train, y_train,
        cv=5, scoring='neg_root_mean_squared_error'
    )

    mlflow.log_metric("catboost_rmse_mean", -catboost_cv_scores.mean())
    mlflow.log_metric("catboost_rmse_std", catboost_cv_scores.std())
    mlflow.log_metric("xgboost_rmse_mean", -xgboost_cv_scores.mean())
    mlflow.log_metric("xgboost_rmse_std", xgboost_cv_scores.std())

    catboost_pred = catboost_model.predict(test_df)
    xgboost_pred = xgboost_model.predict(test_df)
    ensemble_pred = (catboost_pred + xgboost_pred) / 2

    joblib.dump(catboost_model, catboost_model_file)
    joblib.dump(xgboost_model, xgboost_model_file)

    mlflow.sklearn.log_model(catboost_model, "catboost_model")
    mlflow.sklearn.log_model(xgboost_model, "xgboost_model")

    results_df = pd.DataFrame({
        'flight_id': test_df['flight_id'],
        'tow': ensemble_pred
    })
    results_df.to_csv(predictions_file, index=False)

    metrics_df = pd.DataFrame({
        'metric': [
            'CatBoost CV RMSE (mean)',
            'CatBoost CV RMSE (std)',
            'XGBoost CV RMSE (mean)',
            'XGBoost CV RMSE (std)'
        ],
        'value': [
            -catboost_cv_scores.mean(),
            catboost_cv_scores.std(),
            -xgboost_cv_scores.mean(),
            xgboost_cv_scores.std()
        ]
    })
    metrics_df.to_csv(metrics_file, index=False)

    mlflow.end_run()


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