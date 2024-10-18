from zenml.config import DockerSettings
from zenml.pipelines import pipeline

from config import Config
from steps.clean_data import clean_data
from steps.feature_engineering import feature_engineering
from steps.ingest_data import ingest_data
from steps.split_data import split_data
from steps.feature_selection import feature_selection
from steps.train_models import train_models
from steps.predict_and_save import predict_and_save

docker_settings = DockerSettings(required_integrations=["MLFLOW"])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    """Defines the training pipeline with feature selection and model selection."""
    config = Config()
    data = ingest_data(config)
    cleaned_data = clean_data(config, data)
    X, y = feature_engineering(config, cleaned_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_selected, X_test_selected = feature_selection(
        config, X_train, y_train, X_test
    )
    # best_model_info = select_best_model(X_train_selected, y_train, config)
    trained_models = train_models(X_train_selected, y_train, X_test_selected, y_test)
    predict_and_save(trained_models, X_test_selected)
