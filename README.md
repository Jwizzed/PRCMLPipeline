# PRCMLPipeline
End-to-End PRC Data Challenge Pipeline

## Zenml

### Installation
**Required Python:** 3.9

bash
pip install "zenml[server]"
zenml init
zenml up


#### Install MLflow Integration
```
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

#### Start MLflow UI
```
mlflow ui --backend-store-uri "file:/Users/krittinsetdhavanich/Library/Application Support/zenml/local_stores/236ebb32-3647-47d2-a01b-db56888775bb/mlruns"
```

## KubeFlow

### Local Installation

#### Install Prerequisite Tools
```
brew install kubectl  # Command-line tool for interacting with Kubernetes clusters.
brew install --cask multipass  # Allows you to create and manage lightweight virtual machines (VMs) on your local machine.
brew install kustomize  # Used to customize Kubernetes configurations.
brew install git
```

#### Set Up a Local Kubernetes Cluster with k3s
k3s (Minikube alternative). Thanks to [Dev.to](https://dev.to/chillaranand/local-kubernetes-cluster-with-k3s-on-mac-m1-i57).

```
multipass launch --name k3s --memory 12G --disk 40G --cpus 6
multipass shell k3s
curl -sfL https://get.k3s.io | sh -
sudo systemctl status k3s
sudo k3s kubectl get nodes
alias kubectl='sudo k3s kubectl'
exit
```

#### Configure kubectl
```
multipass exec k3s -- sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config  # Copy the kubeconfig file to your local machine.
multipass list  # Modify the server line in the ~/.kube/config file to point to the correct IP address of the VM.
```

Open the `~/.kube/config` file in a text editor and change the server line to:
yaml
server: https://<IP_ADDRESS>:6443


#### Install Kubeflow on the Kubernetes Cluster
Clone the Kubeflow Manifests Repository:
```
git clone https://github.com/kubeflow/manifests.git
cd manifests
```

**Install:**
```
while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
```

**Uninstall:**
```
while ! kustomize build example | awk '!/well-defined/' | kubectl delete -f -; do echo "Retrying to apply resources"; sleep 10; done
```

**Default Account:**
Username: user@example.com
Password: 12341234


#### Access the Kubeflow Dashboard
```
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

In case of an error, change the port to 9091:
```
kubectl edit deployment -n kubeflow workflow-controller
```

If there is any crash or error, wait and try again.

### Manage k3s
**Pause:**
```
multipass pause k3s
multipass start k3s
multipass restart k3s
```

**Delete:**
```
multipass delete k3s
multipass purge
```