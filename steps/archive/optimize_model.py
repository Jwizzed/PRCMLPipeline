from typing import Dict

import pandas as pd
import mlflow
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
import optuna
from sklearn.model_selection import cross_val_score, KFold

from config import Config

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def optimize_model(
    X_train: Annotated[pd.DataFrame, "Train features"],
    y_train: Annotated[pd.Series, "Train labels"],
    config: Config,
) -> Annotated[Dict, "Best hyperparameters"]:
    """Optimizes the model using Optuna and logs results to MLflow."""

    mlflow.autolog(disable=True)

    def objective(trial):
        model_name = config.OPTIMIZATION_MODEL
        if model_name == "XGBoost":
            from xgboost import XGBRegressor

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
            }
            model = XGBRegressor(**params)
        elif model_name == "CatBoost":
            from catboost import CatBoostRegressor

            params = {
                "iterations": trial.suggest_int("iterations", 500, 2000, step=100),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "random_state": 42,
                "verbose": False,
            }
            model = CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv
        )
        rmse = -scores.mean()
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=config.N_TRIALS,
        timeout=config.TIMEOUT,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_score = study.best_value

    with mlflow.start_run(run_name="Model Optimization", nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_rmse", best_score)
        mlflow.set_tag("optimization_model", config.OPTIMIZATION_MODEL)

        study_df = study.trials_dataframe()
        study_csv = "optuna_study.csv"
        study_df.to_csv(study_csv, index=False)
        mlflow.log_artifact(study_csv)

    return best_params
