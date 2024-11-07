from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image='gcr.io/prc-data-pipeline/ml-image',
    packages_to_install=["pandas", "scikit-learn", "xgboost", "joblib"])
def train_xgboost_model(
    x_train_file: InputPath("CSV"),
    y_train_file: InputPath("CSV"),
    x_test_file: InputPath("CSV"),
    y_test_file: InputPath("CSV"),
    model_output: OutputPath("PKL"),
    model_name: str,
    find_best_parameters: bool = False,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    n_estimators: int = 2000,
) -> dict:
    """Trains an XGBoost model and returns predictions and metrics."""
    import pandas as pd
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import joblib

    X_train = pd.read_csv(x_train_file)
    y_train = pd.read_csv(y_train_file)
    X_test = pd.read_csv(x_test_file)
    y_test = pd.read_csv(y_test_file)

    if find_best_parameters:
        model = XGBRegressor(random_state=42)
        param_grid = {
            "n_estimators": [1000, 2000],
            "learning_rate": [0.01, 0.1],
            "max_depth": [4, 6, 8],
        }

        grid_search = GridSearchCV(
            model, param_grid, scoring="neg_mean_squared_error", cv=3, verbose=100
        )
        grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        best_params = grid_search.best_params_

    else:
        model = XGBRegressor(
            random_state=42,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        y_pred = model.predict(X_test)
        best_params = None

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    joblib.dump(model, model_output)

    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return {
        "model_name": model_name,
        "rmse": rmse,
        "best_parameters": best_params,
        "top_features": feature_importance.head(10).to_dict(),
        "predictions": y_pred.tolist(),
    }
