from zenml.config import DockerSettings
from zenml.pipelines import pipeline

from steps.clean_data import clean_data
from steps.evaluation import evaluate_models
from steps.feature_engineering import feature_engineering
from steps.ingest_data import ingest_data
from steps.split_data import split_data
from steps.train_model import train_models

docker_settings = DockerSettings(required_integrations=["catboost"])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    """Defines the training pipeline."""
    trajectory_df, train_df, test_df = ingest_data()
    trajectory_df_cleaned, train_df_cleaned, test_df_cleaned = clean_data(
        trajectory_df, train_df, test_df
    )
    X, y = feature_engineering(trajectory_df_cleaned, train_df_cleaned, test_df_cleaned)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_models(X_train, y_train, X_test, y_test)
    evaluate_models(model, X_test, y_test)
