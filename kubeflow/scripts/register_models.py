from google.cloud import aiplatform
import os

aiplatform.init(project="prc-data-pipeline", location="us-central1")

bucket_path = "gs://prc-data-pipeline/models/"

serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"

models = [
    "b77w_catboost.joblib",
    "b77w_xgboost.joblib",
    "high_catboost.joblib",
    "high_xgboost.joblib",
    "low_catboost.joblib",
    "low_xgboost.joblib",
    "medium_catboost.joblib",
    "medium_xgboost.joblib",
    "non_b77_catboost.joblib",
    "non_b77_xgboost.joblib",
    "very_low_catboost.joblib",
    "very_low_xgboost.joblib",
]

for model_file in models:
    artifact_uri = os.path.join(bucket_path, model_file)
    display_name = model_file.replace(".joblib", "")

    print(f"Registering and deploying model: {display_name}")

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
    )

    endpoint = aiplatform.Endpoint.create(display_name=f"{display_name}_endpoint")

    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{display_name}_deployment",
        machine_type="n1-standard-4",
    )

    print(f"Model {display_name} deployed to endpoint {endpoint.resource_name}")
print("All models registered and deployed successfully.")
