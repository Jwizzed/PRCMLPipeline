from google.cloud import aiplatform
from kfp import compiler

from pipelines.flight_tow_pipeline import flight_tow_pipeline


def run_pipeline(
    project_id: str,
    pipeline_root: str,
    data_path: str,
    location: str = "us-central1",
    pipeline_name: str = "flight-tow-prediction",
):
    compiler.Compiler().compile(
        pipeline_func=flight_tow_pipeline, package_path="flight_tow_pipeline.json"
    )

    aiplatform.init(
        project=project_id,
        location=location,
    )

    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path="flight_tow_pipeline.json",
        pipeline_root=pipeline_root,
        parameter_values={
            "data_path": data_path,
        },
    )

    job.submit()


if __name__ == "__main__":
    PROJECT_ID = "prc-data-pipeline"
    PIPELINE_ROOT = "gs://prc-data-pipeline/pipeline"
    DATA_PATH = "gs://prc-data-pipeline/data"

    run_pipeline(
        project_id=PROJECT_ID,
        pipeline_root=PIPELINE_ROOT,
        data_path=DATA_PATH,
    )
