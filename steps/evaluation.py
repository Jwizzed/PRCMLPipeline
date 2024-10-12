from typing import Any

import pandas as pd
from typing_extensions import Annotated
from zenml import step


@step
def evaluate_model(
    model: Annotated[Any, "Trained model"],
    X_test: Annotated[pd.DataFrame, "Test features"],
    y_test: Annotated[pd.Series, "Test labels"],
) -> None:
    """Evaluates the model performance and prints feature importance."""
    """Evaluates the model performance and prints feature importance."""
    from sklearn.metrics import mean_squared_error
    import numpy as np

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse}")

    feature_importance = pd.DataFrame(
        {"feature": X_test.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
