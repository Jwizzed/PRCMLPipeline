from typing import Any

import pandas as pd
from zenml import step


@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series) -> Any:
    """Trains the CatBoost regression model."""
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6,
                              random_state=42)
    model.fit(X_train, y_train, eval_set=(X_test, y_test),
              early_stopping_rounds=50, verbose=100)
    return model
