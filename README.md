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
brew install --cask multipass  # allows you to create and manage lightweight virtual machines (VMs) on your local machine.
brew install kustomize  # used to customize Kubernetes configurations.
brew install git
```
#### Set Up a Local Kubernetes Cluster with k3s
k3s (Minikube alternative). Thanks to https://dev.to/chillaranand/local-kubernetes-cluster-with-k3s-on-mac-m1-i57
```angular2html
multipass launch --name k3s --mem 4G --disk 40G
multipass shell k3s
curl -sfL https://get.k3s.io | sh -
sudo systemctl status k3s
sudo k3s kubectl get nodes
alias kubectl='sudo k3s kubectl'
exit
```
```angular2html
multipass exec k3s -- sudo cat /etc/rancher/k3s/k3s.yaml > ~/.kube/config  # copy the kubeconfig file to your local machine so that you can use kubectl from your local terminal. This allows you to manage your k3s cluster without needing to SSH into the VM every time.
multipass list  # You may need to modify the server line in the ~/.kube/config file to point to the correct IP address of the VM.
```
Open the ~/.kube/config file in a text editor and change the server line to:
```angular2html
server: https://<IP_ADDRESS>:6443
```
#### Install Kubeflow on the Kubernetes Cluster
Clone the Kubeflow Manifests Repository
```angular2html
git clone https://github.com/kubeflow/manifests.git
cd manifests
```
Install
```
while ! kustomize build example | awk '!/well-defined/' | sudo k3s kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done

```
Uninstall
```
while ! kustomize build example | awk '!/well-defined/' | sudo k3s kubectl delete -f -; do echo "Retrying to apply resources"; sleep 10; done
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

Pause
```angular2html
multipass pause k3s
multipass start k3s
multipass restart k3s
```

Delete
```angular2html
multipass delete k3s
multipass purge
```