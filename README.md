# PRCMLPipeline

End-to-End PRC Data Challenge Pipeline

## Overview

The PRC Data Challenge aims to engage data scientists, even those without an aviation background, to create teams and compete in building an open Machine Learning (ML) model capable of accurately inferring the Actual TakeOff Weight (ATOW) of a flown flight.
![Pipeline Overview](/preview_imgs/pipeline_overview.png)

### The Data

We provide information for 369,013 flights flown across Europe in 2022. The flight data includes:
- Origin/destination airport
- Aircraft type
- Off-block and arrival times
- Estimated TakeOff Weight (TOW)

Through collaboration with OpenSky Network (OSN), we added relevant trajectories at (max) 1-second granularity, accounting for around 158 GiB of parquet files.

The dataset is split into:
- 105,959 flights in `submission_set.csv` for ranking intermediate submissions
- Additional 52,190 flights for final prize ranking

## Pipeline Overview

The pipeline is implemented using Kubeflow on Vertex AI and is structured to handle:
1. Data Ingestion: Load flight and trajectory data
2. Data Preprocessing: Clean, encode, and normalize data
3. Model Training: Train ensemble models using different strategies
4. Model Deployment: Deploy models to Google Cloud Storage

## Installation Options

### Prerequisites
- Python 3.9
- Google Cloud SDK
- Docker
- Kubeflow Pipelines SDK

### 1. Local Kubeflow Installation

#### Install Prerequisite Tools
```bash
brew install kubectl
brew install --cask multipass
brew install kustomize
brew install git
```

#### Set Up Local Kubernetes Cluster with k3s
```bash
multipass launch --name k3s --memory 12G --disk 40G --cpus 6
multipass shell k3s
curl -sfL https://get.k3s.io | sh -
sudo systemctl status k3s
sudo k3s kubectl get nodes
alias kubectl='sudo k3s kubectl'
exit
```

#### Configure kubectl
```bash
multipass exec k3s -- sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config
multipass list  # Get the IP address
```
Then modify `~/.kube/config` to update server line:
```yaml
server: https://<IP_ADDRESS>:6443
```

#### Install Kubeflow
```bash
git clone https://github.com/kubeflow/manifests.git
cd manifests
while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
```

#### Access Dashboard
```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```
Default credentials:
- Username: user@example.com
- Password: 12341234

#### Manage k3s Cluster
```bash
# Pause/Resume
multipass pause k3s
multipass start k3s
multipass restart k3s

# Delete
multipass delete k3s
multipass purge
```

### 2. ZenML Installation

#### Basic Setup
```bash
pip install "zenml[server]"
zenml init
zenml up
```

#### MLflow Integration
```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

#### Start MLflow UI
```bash
mlflow ui --backend-store-uri "file:/Users/<username>/Library/Application Support/zenml/local_stores/<store-id>/mlruns"
```

### 3. Google Cloud Kubeflow Installation

#### Set Up Google Cloud Project
1. Create a new project in Google Cloud Console
2. Enable required APIs:
   - Vertex AI API
   - Kubernetes Engine API
   - Container Registry API
   - Cloud Storage API

#### Configure Google Cloud CLI
```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud services enable aiplatform.googleapis.com
```

#### Deploy Pipeline
Create a pipeline configuration file with:
- Project ID
- Pipeline root location in GCS
- Data path
- Deployment path
- External data path

## Project Structure
```
|- run_pipeline.py           # Main pipeline execution script
|- pipelines
|  |- flight_tow_pipeline.py # Pipeline definition
|- components               # Pipeline components
|  |- data_ingestion
|  |- data_preprocessing
|  |- deployment
|  |- modelling
|- src                     # Source code
```

## Component Requirements

### Dockerfile for Components
Each component requires a Dockerfile:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY kubeflow /app/kubeflow
COPY kubeflow/requirements.txt .
RUN pip install -r requirements.txt
```

### PySpark
```
brew install java
```
```
brew install apache-spark
```
```angular2html
export SPARK_HOME=/opt/homebrew/Cellar/apache-spark/3.5.0/libexec
export PATH=$SPARK_HOME/bin:$PATH
```

### Deployment Strategy

Models are automatically deployed to Google Cloud Storage for inference, eliminating the need for local downloads. The deployment process handles:
- Model versioning
- Storage management
- Access control
- Deployment logging

## Dashboard Access
- Kubeflow dashboard: Access through port forwarding (default: 8080, fallback: 9091)
- MLflow UI: Access for experiment tracking and monitoring
- Google Cloud Console: For Vertex AI pipeline monitoring

## Troubleshooting Tips

1. If Kubeflow dashboard port 8080 fails:
```bash
kubectl edit deployment -n kubeflow workflow-controller
```
Change to port 9091

2. For k3s crash recovery:
- Wait a few minutes
- Restart the service:
```bash
multipass restart k3s
```

3. For MLflow connection issues:
- Verify store ID in the backend URI
- Check port availability
- Ensure proper permissions
