import pandas as pd
import mlflow
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from typing import Dict, Any

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def predict_and_save(
        models: Annotated[Dict[str, Any], "Trained models"],
        X_test: Annotated[pd.DataFrame, "Test features"],
) -> None:
    """Predicts using trained models and saves results to CSV."""

    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_test)

    results_df = pd.DataFrame({'flight_id': X_test.index})

    for model_name, preds in predictions.items():
        results_df[f'tow_{model_name}'] = preds

    csv_path = "results/team_nice_jacket_v5_43c9f53d-3900-4d9a-b19f-42b1c388ca71.csv"
    results_df.to_csv(csv_path, index=False)

    with mlflow.start_run(run_name="predictions_logging", nested=True):
        mlflow.log_artifact(csv_path)

    print(f"Predictions saved to {csv_path}")