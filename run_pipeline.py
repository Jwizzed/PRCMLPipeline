import click
from datetime import datetime
from zenml.client import Client
from pipelines.train_pipeline import train_pipeline


@click.command()
@click.option(
    "--name",
    default=f"Flight Prediction Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    help="Name for this pipeline run",
)
def main(name):
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    pipeline = train_pipeline()
    pipeline.run(run_name=name)


if __name__ == "__main__":
    main()
