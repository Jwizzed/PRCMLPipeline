from kfp.v2.dsl import component, OutputPath, InputPath


@component(
    base_image="gcr.io/prc-data-pipeline/ml-image",
    packages_to_install=["joblib", "google-cloud-storage"],
)
def deploy_model(
    model_file: InputPath("PKL"),
    model_name: str,
    deployment_path: str,
    gcs_path: OutputPath(str),
):
    """Deploys the model to Google Cloud Storage."""
    from google.cloud import storage

    if deployment_path.startswith("gs://"):
        deployment_path = deployment_path[5:]

    bucket_name, folder_path = deployment_path.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if not folder_path.endswith("/"):
        folder_path += "/"

    destination_blob_name = f"{folder_path}{model_name}.joblib"
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(model_file)

    gcs_url = f"gs://{bucket_name}/{destination_blob_name}"
    with open(gcs_path, "w") as f:
        f.write(gcs_url)
