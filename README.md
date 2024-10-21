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
minikube start --cpus=6 --memory=15360 --addons=ingress
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
Install
```
while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
```
Uninstall
```
while ! kustomize build example | awk '!/well-defined/' | kubectl delete -f -; do echo "Retrying to apply resources"; sleep 10; done

```
Default account
```
user@example.com
12341234
```

Access the Kubeflow dashboard
```angular2html
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```
In case an error occurs change port to 9091
```angular2html
kubectl edit deployment -n kubeflow workflow-controller
```
If there is any crash or an error occurs -> wait and try it again.