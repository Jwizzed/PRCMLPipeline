import logging
from typing import Any, Dict
from typing_extensions import Annotated

import mlflow
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_models(
        X_train: Annotated[pd.DataFrame, "Train features"],
        y_train: Annotated[pd.Series, "Train labels"],
        X_test: Annotated[pd.DataFrame, "Test features"],
        y_test: Annotated[pd.Series, "Test labels"],
) -> Annotated[Dict[str, Any], "Trained models"]:
    """Trains multiple regression models and logs them with MLflow."""

    with mlflow.start_run(run_name="data_logging", nested=True):
        mlflow.log_table(data=X_train, artifact_file="X_train.parquet")
        mlflow.log_table(
            data=pd.DataFrame(y_train, columns=["target"]),
            artifact_file="y_train.parquet",
        )

        mlflow.log_table(data=X_test, artifact_file="X_test.parquet")
        mlflow.log_table(
            data=pd.DataFrame(y_test, columns=["target"]),
            artifact_file="y_test.parquet",
        )
        artifact_uri = mlflow.get_artifact_uri()
        mlflow.log_param("artifact_uri", artifact_uri)

    models = {
        "CatBoost": CatBoostRegressor(
            iterations=1000, learning_rate=0.1, depth=6, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=42
        ),
        # "RandomForest": RandomForestRegressor(
        #     n_estimators=1000, max_depth=6, random_state=42
        # ),
        # "LightGBM": LGBMRegressor(
        #     n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=42
        # ),
        # "RandomForest": RandomForestRegressor(
        #     n_estimators=1000, max_depth=6, random_state=42
        # ),
        # "LinearRegression": LinearRegression(),
        # "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        # "GradientBoosting": GradientBoostingRegressor(
        #     n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=42
        # ),

    }

    trained_models = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_training", nested=True):
            if model_name == "CatBoost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_test, y_test),
                    early_stopping_rounds=50,
                    verbose=100,
                )
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_params(model.get_params())

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            input_example = X_test.iloc[[0]]

            if model_name == "CatBoost":
                mlflow.catboost.log_model(model, f"{model_name}_model",
                                          input_example=input_example)
            elif model_name == "XGBoost":
                mlflow.xgboost.log_model(model, f"{model_name}_model",
                                         input_example=input_example)
            else:
                mlflow.sklearn.log_model(model, f"{model_name}_model",
                                         input_example=input_example)

            trained_models[model_name] = model

            logging.info(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return trained_models
