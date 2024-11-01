from kfp.v2.dsl import component, InputPath, OutputPath


@component(packages_to_install=["pandas", "scikit-learn", "catboost", "joblib"])
def train_catboost_model(
    x_train_file: InputPath("CSV"),
    y_train_file: InputPath("CSV"),
    x_test_file: InputPath("CSV"),
    y_test_file: InputPath("CSV"),
    model_output: OutputPath("PKL"),
    model_name: str,
    find_best_parameters: bool = False,
    depth: int = 9,
    iterations: int = 2000,
    learning_rate: float = 0.15,
) -> dict:
    """Trains a CatBoost model and returns predictions and metrics."""
    import pandas as pd
    import numpy as np
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import joblib

    # Load data
    X_train = pd.read_csv(x_train_file)
    y_train = pd.read_csv(y_train_file)
    X_test = pd.read_csv(x_test_file)
    y_test = pd.read_csv(y_test_file)

    if find_best_parameters:
        model = CatBoostRegressor(random_state=42)
        param_grid = {
            "iterations": [1000, 2000],
            "learning_rate": [0.01, 0.1],
            "depth": [4, 6, 8],
        }

        grid_search = GridSearchCV(
            model, param_grid, scoring="neg_mean_squared_error", cv=3, verbose=100
        )
        grid_search.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=100,
        )

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        best_params = grid_search.best_params_

    else:
        model = CatBoostRegressor(
            random_state=42,
            depth=depth,
            iterations=iterations,
            learning_rate=learning_rate,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)
        y_pred = model.predict(X_test)
        best_params = None

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Save model
    joblib.dump(model, model_output)

    # Feature importance
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
