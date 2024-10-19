# PRCMLPipeline
End-to-End PRC Data Challenge Pipeline
## Zenml
### Installation
required python 3.9
```
pip install "zenml["server"]"
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

## KubeFlow
### Installation
#### Install Prerequisite Tools
```angular2html
brew install kubectl  # command-line tool for interacting with Kubernetes clusters.
brew install minikube  # allows you to run a single-node Kubernetes cluster locally.
brew install kustomize  # used to customize Kubernetes configurations.
brew install git
```
#### Set Up a Local Kubernetes Cluster with Minikube
Start Minikube
```angular2html
minikube start --cpus=4 --memory=8192 --addons=ingress
```
kubectl cluster-info
```angular2html
kubectl cluster-info
```
#### Install Kubeflow on the Kubernetes Cluster
Clone the Kubeflow Manifests Repository
```angular2html
git clone https://github.com/kubeflow/manifests.git
cd manifests
```
```angular2html
Install Kubeflow
# Cert-manager
kustomize build common/cert-manager/cert-manager/base | kubectl apply -f -
kubectl wait --for=condition=ready pod -l 'app in (cert-manager,webhook)' --timeout=180s -n cert-manager
kustomize build common/cert-manager/kubeflow-issuer/base | kubectl apply -f -

# Istio
kustomize build common/istio-1-22/istio-crds/base | kubectl apply -f -
kustomize build common/istio-1-22/istio-namespace/base | kubectl apply -f -
kustomize build common/istio-1-22/istio-install/base | kubectl apply -f -

# Dex
kustomize build common/dex/overlays/istio | kubectl apply -f -

# OIDC AuthService
kustomize build common/oidc-authservice/base | kubectl apply -f -

# Knative
kustomize build common/knative/knative-serving/overlays/gateways | kubectl apply -f -
kustomize build common/istio-1-22/cluster-local-gateway/base | kubectl apply -f -

# Kubeflow Namespace
kustomize build common/kubeflow-namespace/base | kubectl apply -f -

# Kubeflow Roles
kustomize build common/kubeflow-roles/base | kubectl apply -f -

# Kubeflow Istio Resources
kustomize build common/istio-1-22/kubeflow-istio-resources/base | kubectl apply -f -

# Kubeflow Pipelines
kustomize build apps/pipeline/upstream/env/platform-agnostic-multi-user | kubectl apply -f -

# KServe
kustomize build contrib/kserve/kserve | kubectl apply -f -
kustomize build contrib/kserve/models-web-app/overlays/kubeflow | kubectl apply -f -

# Katib
kustomize build apps/katib/upstream/installs/katib-with-kubeflow | kubectl apply -f -

# Central Dashboard
kustomize build apps/centraldashboard/upstream/overlays/kserve | kubectl apply -f -

# Admission Webhook
kustomize build apps/admission-webhook/upstream/overlays/cert-manager | kubectl apply -f -

# Notebooks
kustomize build apps/jupyter/notebook-controller/upstream/overlays/kubeflow | kubectl apply -f -
kustomize build apps/jupyter/jupyter-web-app/upstream/overlays/istio | kubectl apply -f -

# Profiles + KFAM
kustomize build apps/profiles/upstream/overlays/kubeflow | kubectl apply -f -

# Volumes Web App
kustomize build apps/volumes-web-app/upstream/overlays/istio | kubectl apply -f -

# Tensorboard
kustomize build apps/tensorboard/tensorboards-web-app/upstream/overlays/istio | kubectl apply -f -
kustomize build apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow | kubectl apply -f -

# Training Operator
kustomize build apps/training-operator/upstream/overlays/kubeflow | kubectl apply -f -

# User Namespace
kustomize build common/user-namespace/base | kubectl apply -f -
```
Access the Kubeflow dashboard
```angular2html
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```
In case an error occurs change port to 9091
```angular2html
kubectl edit deployment -n kubeflow workflow-controller
```
```angular2html
kubectl delete deployment -n kubeflow workflow-controller
kustomize build apps/pipeline/upstream/env/platform-agnostic-multi-user | kubectl apply -f -
```