import logging
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import mean_squared_error, r2_score
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_models(
    models: Annotated[Dict[str, Any], "Trained models"],
    X_test: Annotated[pd.DataFrame, "Test features"],
    y_test: Annotated[pd.Series, "Test labels"],
) -> None:
    """Evaluates the performance of multiple models and logs feature importance."""
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_evaluation", nested=True):
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            logging.info(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

            if hasattr(model, "feature_importances_"):
                feature_importance = pd.DataFrame(
                    {
                        "feature": X_test.columns,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)

                logging.info(f"\nTop 10 Most Important Features for {model_name}:")
                for _, row in feature_importance.head(10).iterrows():
                    logging.info(f"{row['feature']}: {row['importance']:.4f}")

                fig = go.Figure(
                    go.Bar(
                        x=feature_importance["importance"][:10],
                        y=feature_importance["feature"][:10],
                        orientation="h",
                    )
                )

                fig.update_layout(
                    title=f"Top 10 Feature Importances - {model_name}",
                    xaxis_title="Importance",
                    yaxis_title="Features",
                    yaxis=dict(autorange="reversed"),
                )

                png_path = (
                    f"assets/importance_features/{model_name}_feature_importance.png"
                )
                pio.write_image(fig, png_path)
                mlflow.log_artifact(png_path)

            else:
                logging.info(
                    f"{model_name} does not have feature_importances_ attribute"
                )
