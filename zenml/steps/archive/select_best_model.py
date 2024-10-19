# from typing import Dict
# from zenml import step
# from typing_extensions import Annotated
# from zenml.client import Client
# from config import Config
# import mlflow
# import pandas as pd
#
# # from pycaret.regression import *
#
# experiment_tracker = Client().active_stack.experiment_tracker
#
#
# @step(experiment_tracker=experiment_tracker.name)
# def select_best_model(
#     X_train: Annotated[pd.DataFrame, "Train features"],
#     y_train: Annotated[pd.DataFrame, "Train labels"],
#     config: Config,
# ) -> Dict[str, any]:
#     """Selects the best model using PyCaret's automated model selection."""
#
#     mlflow.autolog(disable=True)
#     train_data = pd.concat([X_train, y_train], axis=1)
#     reg = setup(data=train_data, target="tow", session_id=42)
#     best_model = compare_models(n_select=1, sort="RMSE")
#
#     best_model_name = best_model.__class__.__name__
#     best_model_params = best_model.get_params()
#
#     with mlflow.start_run(run_name="Model Selection with PyCaret", nested=True):
#         mlflow.log_param("best_model", best_model_name)
#         mlflow.log_params(best_model_params)
#
#     return {
#         "model": best_model,
#         "model_name": best_model_name,
#         "params": best_model_params,
#     }
