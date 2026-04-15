# Troubleshoot and Deploy ArgoCD Application

Write-Host "Checking Kubernetes cluster status..." -ForegroundColor Cyan

# Check if kubectl is available
$kubectlVersion = kubectl version --client --short 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "kubectl not found or not configured properly" -ForegroundColor Red
    exit 1
}
Write-Host "kubectl version: $kubectlVersion" -ForegroundColor Green

# Check cluster connectivity
Write-Host "`nChecking cluster connection..." -ForegroundColor Cyan
$clusterInfo = kubectl cluster-info 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Cannot connect to Kubernetes cluster. Ensure minikube is running." -ForegroundColor Red
    Write-Host "Try: minikube start --driver=docker" -ForegroundColor Yellow
    exit 1
}
Write-Host "Connected to cluster" -ForegroundColor Green

# Check if argocd namespace exists
Write-Host "`nChecking ArgoCD namespace..." -ForegroundColor Cyan
$argocdNs = kubectl get namespace argocd --no-headers 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ArgoCD namespace not found. Creating..." -ForegroundColor Yellow
    kubectl create namespace argocd
}
Write-Host "ArgoCD namespace ready" -ForegroundColor Green

# Check ArgoCD server deployment
Write-Host "`nChecking ArgoCD server deployment..." -ForegroundColor Cyan
$argocdServer = kubectl get deployment argocd-server -n argocd --no-headers 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ArgoCD server not deployed. Installing..." -ForegroundColor Yellow
    kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
    Write-Host "Waiting for ArgoCD server to be ready..." -ForegroundColor Cyan
    kubectl wait --for=condition=available --timeout=600s deployment/argocd-server -n argocd
}
Write-Host "ArgoCD server is running" -ForegroundColor Green

# Check if rag-api namespace exists
Write-Host "`nChecking rag-api namespace..." -ForegroundColor Cyan
$ragApiNs = kubectl get namespace rag-api --no-headers 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "rag-api namespace already exists" -ForegroundColor Green
} else {
    Write-Host "Creating rag-api namespace..." -ForegroundColor Yellow
    kubectl create namespace rag-api
}

# Validate YAML
Write-Host "`nValidating ArgoCD Application YAML..." -ForegroundColor Cyan
$yamlPath = "deploy/argocd/rag-api-application.yaml"
$validation = kubectl apply -f $yamlPath --dry-run=client 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "YAML validation failed:" -ForegroundColor Red
    Write-Host $validation
    exit 1
}
Write-Host "YAML is valid" -ForegroundColor Green

# Apply the application
Write-Host "`nApplying ArgoCD Application..." -ForegroundColor Cyan
kubectl apply -f $yamlPath
if ($LASTEXITCODE -eq 0) {
    Write-Host "Application applied successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to apply application" -ForegroundColor Red
    exit 1
}

# Check application status
Write-Host "`nApplication Details:" -ForegroundColor Cyan
Start-Sleep -Seconds 2
kubectl get applications -n argocd -o wide
Write-Host "`n"
kubectl describe application rag-api -n argocd 2>$null || Write-Host "Application still initializing..." -ForegroundColor Yellow

Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "1. Port-forward to ArgoCD UI: kubectl port-forward svc/argocd-server -n argocd 8080:443"
Write-Host "2. Get admin password: kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath='{.data.password}' | base64 --decode"
Write-Host "3. Check rag-api pods: kubectl get pods -n rag-api"
Write-Host "4. View rag-api logs: kubectl logs -n rag-api -l app=rag-api --tail=50 -f"
