from typing import Tuple

import mlflow
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from sklearn.feature_selection import SelectKBest, f_regression

from config import Config

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def feature_selection(
    config: Config,
    X_train: Annotated[pd.DataFrame, "Train features"],
    y_train: Annotated[pd.Series, "Train labels"],
    X_test: Annotated[pd.DataFrame, "Test features"],
) -> Tuple[
    Annotated[pd.DataFrame, "Selected Train features"],
    Annotated[pd.DataFrame, "Selected Test features"],
]:
    """Selects the best features from the training data and applies them to the test data."""
    k = config.NUM_FEATURES
    with mlflow.start_run(run_name="Feature Selection", nested=True):
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()]
        selected_features_df = pd.DataFrame(
            selected_features, columns=["Selected Features"]
        )
        selected_features_csv = "results/selected_features.csv"
        selected_features_df.to_csv(selected_features_csv, index=False)
        mlflow.log_artifact(selected_features_csv)

        feature_scores_df = pd.DataFrame(
            {
                "Feature": X_train.columns,
                "Score": selector.scores_,
                "p-value": selector.pvalues_,
            }
        )
        feature_scores_csv = "results/feature_scores.csv"
        feature_scores_df.to_csv(feature_scores_csv, index=False)
        mlflow.log_artifact(feature_scores_csv)

        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        X_train_selected = pd.DataFrame(
            X_train_selected, columns=selected_features, index=X_train.index
        )
        X_test_selected = pd.DataFrame(
            X_test_selected, columns=selected_features, index=X_test.index
        )

    return X_train_selected, X_test_selected
