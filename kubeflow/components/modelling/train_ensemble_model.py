from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image='gcr.io/prc-data-pipeline/ml-image',
    packages_to_install=["pandas", "scikit-learn", "catboost", "xgboost", "joblib"]
)
def train_ensemble_model(
    x_train_file: InputPath("CSV"),
    y_train_file: InputPath("CSV"),
    x_test_file: InputPath("CSV"),
    y_test_file: InputPath("CSV"),
    catboost_model_output: OutputPath("PKL"),
    xgboost_model_output: OutputPath("PKL"),
    model_name: str,
    find_best_parameters: bool = False,
    catboost_depth: int = 9,
    catboost_iterations: int = 2000,
    catboost_learning_rate: float = 0.15,
    xgboost_learning_rate: float = 0.05,
    xgboost_max_depth: int = 8,
    xgboost_n_estimators: int = 2000,
) -> dict:
    """Trains both CatBoost and XGBoost models and creates an ensemble."""
    import pandas as pd
    import numpy as np
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import joblib

    X_train = pd.read_csv(x_train_file)
    y_train = pd.read_csv(y_train_file)
    X_test = pd.read_csv(x_test_file)
    y_test = pd.read_csv(y_test_file)

    if find_best_parameters:
        catboost_model = CatBoostRegressor(random_state=42)
        catboost_param_grid = {
            "iterations": [1000, 2000],
            "learning_rate": [0.01, 0.1],
            "depth": [4, 6, 8],
        }

        catboost_grid_search = GridSearchCV(
            catboost_model,
            catboost_param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            verbose=100,
        )
        catboost_grid_search.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=100,
        )

        best_catboost = catboost_grid_search.best_estimator_
        catboost_params = catboost_grid_search.best_params_
    else:
        best_catboost = CatBoostRegressor(
            random_state=42,
            depth=catboost_depth,
            iterations=catboost_iterations,
            learning_rate=catboost_learning_rate,
        )
        best_catboost.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)
        catboost_params = None

    if find_best_parameters:
        xgboost_model = XGBRegressor(random_state=42)
        xgboost_param_grid = {
            "n_estimators": [1000, 2000],
            "learning_rate": [0.01, 0.1],
            "max_depth": [4, 6, 8],
        }

        xgboost_grid_search = GridSearchCV(
            xgboost_model,
            xgboost_param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            verbose=100,
        )
        xgboost_grid_search.fit(
            X_train, y_train, eval_set=[(X_test, y_test)], verbose=100
        )

        best_xgboost = xgboost_grid_search.best_estimator_
        xgboost_params = xgboost_grid_search.best_params_
    else:
        best_xgboost = XGBRegressor(
            random_state=42,
            learning_rate=xgboost_learning_rate,
            max_depth=xgboost_max_depth,
            n_estimators=xgboost_n_estimators,
        )
        best_xgboost.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        xgboost_params = None

    joblib.dump(best_catboost, catboost_model_output)
    joblib.dump(best_xgboost, xgboost_model_output)

    y_pred_catboost = best_catboost.predict(X_test)
    y_pred_xgboost = best_xgboost.predict(X_test)
    y_pred_ensemble = (y_pred_catboost + y_pred_xgboost) / 2

    catboost_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_catboost)))
    xgboost_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_xgboost)))
    ensemble_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_ensemble)))

    catboost_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "catboost_importance": best_catboost.feature_importances_,
        }
    ).set_index("feature")

    xgboost_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "xgboost_importance": best_xgboost.feature_importances_,
        }
    ).set_index("feature")

    feature_importance = pd.concat([catboost_importance, xgboost_importance], axis=1)
    feature_importance["mean_importance"] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values(
        "mean_importance", ascending=False
    )

    return {
        "model_name": model_name,
        "catboost_rmse": catboost_rmse,
        "xgboost_rmse": xgboost_rmse,
        "ensemble_rmse": ensemble_rmse,
        "catboost_parameters": catboost_params,
        "xgboost_parameters": xgboost_params,
        "top_features": feature_importance.head(10).to_dict(),
        "ensemble_predictions": y_pred_ensemble.tolist(),
        "catboost_predictions": y_pred_catboost.tolist(),
        "xgboost_predictions": y_pred_xgboost.tolist(),
    }
