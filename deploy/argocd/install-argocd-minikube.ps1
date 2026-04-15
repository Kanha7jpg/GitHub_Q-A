$ErrorActionPreference = "Stop"

if (-not (Get-Command minikube -ErrorAction SilentlyContinue)) {
    Write-Error "minikube is not installed or not in PATH. Install with: winget install Kubernetes.minikube"
}

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Error "kubectl is not installed or not in PATH. Install with: winget install Kubernetes.kubectl"
}

Write-Host "Starting Minikube..."
minikube start

Write-Host "Installing ArgoCD..."
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --for=condition=available --timeout=600s deployment/argocd-server -n argocd

Write-Host "ArgoCD is ready. Port-forward it in a separate terminal with:"
Write-Host "kubectl port-forward svc/argocd-server -n argocd 8080:443"
Write-Host "Then retrieve the initial admin password with:"
Write-Host "kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath=\"{.data.password}\" | base64 --decode"
