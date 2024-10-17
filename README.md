# PRCMLPipeline
End-to-End PRC Data Challenge Pipeline

# Installation
required python 3.9

pip install "zenml["server"]"

```angular2html
zenml init
zenml up
```
```angular2html
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```
```angular2html
mlflow ui --backend-store-uri "file:/Users/krittinsetdhavanich/Library/Application Support/zenml/local_stores/236ebb32-3647-47d2-a01b-db56888775bb/mlruns"
```