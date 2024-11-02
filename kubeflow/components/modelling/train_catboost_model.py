from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    packages_to_install=[
        "catboost",
        "pandas",
        "scikit-learn",
        "numpy",
        "joblib"
    ]
)
def train_catboost_model(
        X_train_file: InputPath("CSV"),
        y_train_file: InputPath("CSV"),
        X_test_file: InputPath("CSV"),
        y_test_file: InputPath("CSV"),
        model_output: OutputPath("pkl"),
        feature_importance_output: OutputPath("CSV"),
        metrics_output: OutputPath("JSON"),
        model_name: str,
        find_best_parameters: bool = False
):
    """Trains a CatBoost model and saves it along with metrics and feature importance."""
    import pandas as pd
    import numpy as np
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import joblib
    import json

    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file)
    X_test = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file)

    if find_best_parameters:
        model = CatBoostRegressor(random_state=42)
        param_grid = {
            'iterations': [1000, 2000],
            'learning_rate': [0.01, 0.1],
            'depth': [4, 6, 8]
        }

        grid_search = GridSearchCV(model, param_grid,
                                   scoring='neg_mean_squared_error', cv=3,
                                   verbose=100)
        grid_search.fit(X_train, y_train, eval_set=(X_test, y_test),
                        early_stopping_rounds=50, verbose=100)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = CatBoostRegressor(random_state=42, depth=9,
                                       iterations=2000, learning_rate=0.15)
        best_model.fit(X_train, y_train, eval_set=(X_test, y_test),
                       verbose=100)
        best_params = {
            'depth': 9,
            'iterations': 2000,
            'learning_rate': 0.15
        }

    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    joblib.dump(best_model, model_output)

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(feature_importance_output, index=False)

    metrics = {
        'model_name': model_name,
        'rmse': rmse,
        'best_parameters': best_params
    }
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f)