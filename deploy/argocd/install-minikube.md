# ArgoCD On Minikube

## 1. Ensure CLI tools are installed

```bash
winget install Kubernetes.minikube
winget install Kubernetes.kubectl
```

## 2. Start Minikube

```bash
minikube start
```

## 3. Install ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --for=condition=available --timeout=600s deployment/argocd-server -n argocd
```

You can also run the automation script:

```bash
powershell -ExecutionPolicy Bypass -File deploy/argocd/install-argocd-minikube.ps1
```

## 4. Access ArgoCD UI

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Get initial admin password:

```bash
kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath="{.data.password}" | base64 --decode
```

UI URL: https://localhost:8080
User: admin
Password: output of command above

## 5. Register GitLab Repo (if private)

In ArgoCD UI:
1. Settings > Repositories > Connect Repo.
2. Use HTTPS and a GitLab PAT with `read_repository` scope.

## 6. Create Application

Edit `deploy/argocd/rag-api-application.yaml`:
- Replace `repoURL` with your GitLab repo URL.
- Replace `image.repository` with your GitLab Registry runtime image path.
- Optionally change `targetRevision`.

Apply:

```bash
kubectl apply -f deploy/argocd/rag-api-application.yaml
```

## 7. Verify Sync

```bash
kubectl get applications -n argocd
kubectl describe application rag-api -n argocd
kubectl get pods -n rag-api
```
