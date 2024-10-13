from zenml.client import Client
from pipelines.train_pipeline import train_pipeline


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    pipeline = train_pipeline()
    pipeline.run()
