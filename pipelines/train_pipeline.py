from zenml.config import DockerSettings
from zenml.pipelines import pipeline

from config import Config
from steps.clean_data import clean_data
from steps.evaluation import evaluate_models
from steps.feature_engineering import feature_engineering
from steps.ingest_data import ingest_data
from steps.split_data import split_data
from steps.train_model import train_models

docker_settings = DockerSettings(required_integrations=["MLFLOW"])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    """Defines the training pipeline."""
    config = Config()  # Pipeline interface is not allow to pass unknown params.
    data = ingest_data(config)
    cleaned_data = clean_data(config, data)
    X, y = feature_engineering(config, cleaned_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_models(X_train, y_train, X_test, y_test)
    evaluate_models(model, X_test, y_test)
