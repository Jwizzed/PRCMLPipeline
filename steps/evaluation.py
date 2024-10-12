from typing import Any

import pandas as pd
from zenml import step


@step
def evaluate_model(model: Any, X_test: pd.DataFrame,
                   y_test: pd.Series) -> None:
    """Evaluates the model performance and prints feature importance."""
    from sklearn.metrics import mean_squared_error
    import numpy as np
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse}")

    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
