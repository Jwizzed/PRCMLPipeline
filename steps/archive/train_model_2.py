# import logging
# from typing import Any, Dict
#
# import mlflow
# import numpy as np
# import pandas as pd
# from zenml import step
# from zenml.client import Client
# from sklearn.metrics import mean_squared_error, r2_score
# from typing_extensions import Annotated
#
# experiment_tracker = Client().active_stack.experiment_tracker
#
#
# @step(experiment_tracker=experiment_tracker.name)
# def train_models(
#         X_train: Annotated[pd.DataFrame, "Train features"],
#         y_train: Annotated[pd.Series, "Train labels"],
#         X_test: Annotated[pd.DataFrame, "Test features"],
#         y_test: Annotated[pd.Series, "Test labels"],
#         best_model: Any,
#         model_name: str,
# ) -> Annotated[Dict[str, Any], "Trained models"]:
#     """Trains the selected model using optimized hyperparameters and logs it with MLflow."""
#
#     with mlflow.start_run(run_name="data_logging", nested=True):
#         mlflow.log_table(data=X_train, artifact_file="X_train.parquet")
#         mlflow.log_table(
#             data=pd.DataFrame(y_train, columns=["target"]),
#             artifact_file="y_train.parquet",
#         )
#
#         mlflow.log_table(data=X_test, artifact_file="X_test.parquet")
#         mlflow.log_table(
#             data=pd.DataFrame(y_test, columns=["target"]),
#             artifact_file="y_test.parquet",
#         )
#         artifact_uri = mlflow.get_artifact_uri()
#         mlflow.log_param("artifact_uri", artifact_uri)
#
#     best_model.fit(X_train, y_train)
#     y_pred = best_model.predict(X_test)
#
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#
#     mlflow.log_params(best_model.get_params())
#
#     mlflow.log_metric("mse", mse)
#     mlflow.log_metric("rmse", rmse)
#     mlflow.log_metric("r2", r2)
#
#     logging.info(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
#
#     return {model_name: best_model}
