# Cloud-Native GitHub Q&A Assistant (Local-First RAG)

A **production-ready Retrieval-Augmented Generation (RAG)** system that enables local-first question answering over GitHub repositories without any external API dependencies.

🎯 **Key Highlights**:
- ✅ **Local-First**: All models run locally (Phi-3.5-mini-instruct SLM)
- ✅ **Zero API Calls**: No external dependencies, works offline
- ✅ **CPU-Optimized**: Quantized models (3GB instead of 30GB)
- ✅ **Kubernetes-Ready**: Production deployment on Minikube or cloud
- ✅ **GitOps Enabled**: ArgoCD for declarative deployment
- ✅ **Persistent Storage**: ChromaDB vector database with PVC

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed architecture and design decisions.

---

## 📐 Architecture Overview

```
GitHub Repository
    │ (git clone)
    ▼
┌─────────────────────────────────────────┐
│     Code-as-Data Ingestion Pipeline     │
│  Extract .py & .md → Chunk → Embed     │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Embedding Service (all-MiniLM-L6-v2)   │
│  • 384-dimensional vectors              │
│  • Semantic understanding of code       │
│  • Local computation (no API)            │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│   ChromaDB Vector Store (Persistent)    │
│  • SQLite-backed storage                │
│  • HNSW indexing                        │
│  • Cosine similarity search             │
└──────────┬──────────────────────────────┘
           │
           ├─────────────────────┐
           │ Query Time Flow:    │
           │                     │
           ▼                     │
┌─────────────────────────────┐ │
│  User Query → Embedding     │ │
│  Find top 3 similar chunks  │ │
└──────────┬──────────────────┘ │
           │                    │
           ▼                    │
┌──────────────────────────────────────────┐
│    RAG Engine (Augmented Prompt)         │
│  "Context: [snippets] Question: [query]" │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  SLM Inference (Phi-3.5-mini-instruct)  │
│  • Quantized (GGUF) or FP16             │
│  • CPU-only inference                   │
│  • ~3GB memory (quantized)              │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Generated Answer + Source References    │
└──────────────────────────────────────────┘
```

---

## 🚀 Features

- Clones public GitHub repository from `REPO_URL`
- Intelligent ingestion of `.py` and `.md` files only
- **Code-aware chunking** - RecursiveCharacterTextSplitter respects Python syntax
- **Local embeddings** - sentence-transformers (no HuggingFace API calls)
- **Persistent vector storage** - ChromaDB with SQLite backend
- **Top-3 semantic search** - HNSW-indexed retrieval
- **Grounded generation** - Phi-3.5-mini-instruct answers from retrieved context
- **RESTful API**:
  - `GET /health` - System and model status
  - `GET /livez` - Kubernetes liveness probe
  - `GET /readyz` - Kubernetes readiness probe
  - `GET /chunks` - Debug endpoint for ingested data
  - `POST /ask` - Query processing with RAG

## 💾 Resource Requirements

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| **RAM** | 6 GB | 12 GB |
| **CPU Cores** | 2 | 4+ |
| **Disk Space** | 30 GB | 50 GB |
| **GPU** | Not required | Optional (10x speedup) |

**Memory Breakdown**:
```
Phi-3.5-mini (quantized):    ~3.0 GB
Embedding model:             ~1.2 GB
Vector DB + embeddings:      ~1.5 GB
Python runtime + FastAPI:    ~0.5 GB
Kubernetes overhead:         ~0.8 GB
─────────────────────────────
Total:                       ~7.0 GB
```

**Query Latency**:
| Operation | Time |
|-----------|------|
| Embed query | 50-100 ms |
| Vector search (top-3) | 10-50 ms |
| LLM inference | 2-5 seconds |
| **Total** | **~2-5 seconds** |

**Model Load Time (first startup)**:
| Phase | Time |
|-------|------|
| Environment setup | 10s |
| Model download | 2-5 min |
| Model quantization | 5-10 min |
| Repository ingestion | 2-5 min |
| **Total** | **15-25 minutes** |

**Subsequent Restarts**: ~2-3 minutes (from cache)

---

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set at minimum:

```env
REPO_URL=https://github.com/<owner>/<repo>
SLM_MODEL_PATH=C:/path/to/Phi-3.5-mini-instruct
SLM_LOCAL_FILES_ONLY=true
```

Optional GGUF (plug-and-play):

```env
GGUF_MODEL_PATH=C:/path/to/phi-3.5-q4_k_m.gguf
SLM_BACKEND=auto
```

When `GGUF_MODEL_PATH` is present and `llama-cpp-python` is installed, the app uses GGUF automatically.

You can also point `REPO_URL` at a local directory to run the app offline against files in that folder.
If you already downloaded Phi locally, set `SLM_MODEL_PATH` to that folder so startup loads directly from disk instead of reaching out to Hugging Face.
The Hugging Face modules cache defaults to `.cache/hf_modules` (outside `data/`).

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## ✅ Testing

### Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=app

# Generate coverage report
pytest tests/ --cov=app --cov-report=html
```

### Integration Tests

```bash
# Test in local environment
export REPO_URL=https://github.com/Kanha7jpg/BasicLearning
export SLM_LOCAL_FILES_ONLY=true

# Start server
uvicorn app.main:app --reload

# In another terminal, test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/chunks?limit=5
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What Python files are in this repo?"}'
```

### Kubernetes End-to-End Test

```bash
# 1. Deploy
kubectl apply -f deploy/argocd/rag-api-application.yaml

# 2. Wait for ready
kubectl rollout status deployment/rag-api -n rag-api

# 3. Port forward
kubectl port-forward -n rag-api svc/rag-api 8000:8000 &

# 4. Test
sleep 5
curl http://localhost:8000/health
curl http://localhost:8000/readyz
curl http://localhost:8000/livez
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Describe this repository"}'

# 5. Cleanup
kill %1  # Stop port-forward
```

---

## 📊 Monitoring & Logs

```bash
# Follow application logs
kubectl logs -n rag-api -l app=rag-api -f

# Check startup progress
kubectl logs -n rag-api -l app=rag-api | grep "startup:"

# Monitor resource usage
kubectl top pods -n rag-api --containers

# Check pod events
kubectl describe pod -n rag-api -l app=rag-api

# Access application metrics
curl http://localhost:8000/health | jq
```

---

## 📚 Documentation

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Complete architecture and design decisions
- **[APPLICATION_RUN_GUIDE.md](APPLICATION_RUN_GUIDE.md)** - Runtime configuration details
- **[LOCAL_FIRST_ASSESSMENT.md](LOCAL_FIRST_ASSESSMENT.md)** - Verification of local-first design
- **[PRD_IMPLEMENTATION_STATUS.md](PRD_IMPLEMENTATION_STATUS.md)** - PRD requirements mapping
- **[EXECUTION_PLAN.md](EXECUTION_PLAN.md)** - Deployment checklist and timeline
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick lookup guide

---

## 🔐 Security Considerations

### Data Privacy
- ✅ **All models run locally** - No data sent to external services
- ✅ **Persistent storage local** - Vector DB stored in PVC
- ✅ **Offline capable** - Can work without internet after initial setup
- ✅ **No analytics** - No telemetry or tracking

### Kubernetes Security
```yaml
# Consider adding to Helm values for production:
podSecurityPolicy:
  enabled: true
  
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  
resources:
  limits:
    memory: "6Gi"
    cpu: "2000m"
```

### Model Security
- Models downloaded from HuggingFace (trusted source)
- GGUF format provides checksum verification
- Consider air-gapped deployment for regulated environments

---

## 🎓 Learning Resources

### Understanding RAG
- [Retrieval-Augmented Generation (LangChain)](https://python.langchain.com/docs/modules/data_connection/)
- [Vector Embeddings Explained](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Kubernetes & Deployment
- [Helm Charts Documentation](https://helm.sh/docs/)
- [ArgoCD Getting Started](https://argo-cd.readthedocs.io/en/stable/)
- [Kubernetes Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)

### Model Optimization
- [Model Quantization (GGUF)](https://github.com/ggerganov/llama.cpp)
- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [Sentence Transformers](https://www.sbert.net/)

---

## 🚀 Getting Started (Quick Start)

```bash
# 1. Clone repository
git clone https://github.com/Kanha7jpg/GitHub_Q-A.git
cd GitHub_Q-A

# 2. Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure (copy and edit)
cp .env.example .env
# Edit .env with your REPO_URL

# 4. Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Query!
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this repository about?"}'
```

---

## 📞 Troubleshooting Guide

See [DEPLOYMENT_TROUBLESHOOTING.md](deploy/DEPLOYMENT_TROUBLESHOOTING.md) for common issues and solutions.

---

## 📄 License

This project is part of the Cloud-Native GitHub Q&A Assistant initiative.

---

## 📝 Citation

If you use this project in research or production, please cite:

```bibtex
@software{github_q_a_2026,
  author = {Your Name},
  title = {Cloud-Native GitHub Q&A Assistant (Local-First RAG)},
  year = {2026},
  url = {https://github.com/Kanha7jpg/GitHub_Q-A}
}
```

---

**Last Updated**: April 2026  
**Status**: Production Ready ✅

## Containerization

This repo includes a multi-stage Dockerfile that supports two production flows:

1. Preloaded image: downloads Phi-3.5 and quantizes to GGUF during build.
2. Kubernetes initContainer: downloads and quantizes before the app container starts.

### Build preloaded runtime image (fastest startup)

```bash
docker build -t github-q-a:runtime .
```

The default final stage is `runtime-preloaded`, which bakes the GGUF into `/opt/models/phi3_quantized`.

### Build initContainer image

```bash
docker build --target quantizer -t github-q-a:quantizer .
```

The `quantizer` target is meant for initContainer jobs and writes GGUF files to `/models/phi3_quantized`.

### Run container locally

```bash
docker run --rm -p 8000:8000 \
  -e REPO_URL=https://github.com/Kanha7jpg/BasicLearning \
  github-q-a:runtime
```

### Kubernetes with initContainer

Use `deploy/k8s/rag-api-initcontainer.yaml`.

```bash
kubectl apply -f deploy/k8s/rag-api-initcontainer.yaml
```

Update image names in the manifest to your registry tags before applying.

---

## 📦 Kubernetes Deployment (Production)

### Prerequisites
```bash
# 1. Kubernetes cluster (local: Minikube, cloud: EKS/GKE/AKS)
minikube start --driver=docker --memory=8192 --cpus=4

# 2. kubectl configured
kubectl cluster-info

# 3. ArgoCD installed (optional but recommended)
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

### Deploy via Helm

```bash
# Add/update repo (if using external Helm charts)
helm repo add myrepo https://example.com/charts

# Install chart
helm install rag-api deploy/helm/rag-api/ \
  -n rag-api \
  --create-namespace \
  -f deploy/helm/rag-api/values.yaml

# Verify deployment
kubectl get pods -n rag-api
kubectl get svc -n rag-api
```

### Deploy via ArgoCD (GitOps)

```bash
# Apply ArgoCD Application manifest
kubectl apply -f deploy/argocd/rag-api-application.yaml

# Monitor sync status
kubectl get applications -n argocd
kubectl describe application rag-api -n argocd

# Wait for "Synced" and "Healthy" status
kubectl get applications -n argocd -w
```

### Access Application

```bash
# Port forward to application
kubectl port-forward -n rag-api svc/rag-api 8000:8000

# Test health endpoint
curl http://localhost:8000/health

# Send query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What does this repository do?"}'
```

---

## 🔧 Kubernetes Troubleshooting

### Problem: Pod stuck in "Pending" state

**Symptoms**: `kubectl get pods -n rag-api` shows `Pending`

**Causes**:
- Insufficient node resources
- PVC not bound to node
- Image not found

**Solutions**:
```bash
# Check pod events
kubectl describe pod -n rag-api -l app=rag-api

# Check PVC status
kubectl get pvc -n rag-api
kubectl describe pvc -n rag-api

# Check node resources
kubectl top nodes
kubectl describe nodes

# Increase Minikube resources
minikube config set memory 10240
minikube stop
minikube start --driver=docker
```

### Problem: Pod in "CrashLoopBackOff" state

**Symptoms**: Pod repeatedly crashes and restarts

**Causes**:
- Model download failed
- Insufficient memory
- Repository clone failed
- GGUF model missing

**Solutions**:
```bash
# Check logs
kubectl logs -n rag-api -l app=rag-api --tail=100
kubectl logs -n rag-api -l app=rag-api -p  # Previous pod logs

# Check memory usage
kubectl top pods -n rag-api

# If OOMKilled, increase Helm limits
helm upgrade rag-api deploy/helm/rag-api/ \
  -n rag-api \
  --set resources.limits.memory=8Gi \
  --set resources.requests.memory=6Gi

# Verify image exists and is pullable
kubectl run -it --rm debug --image=ubuntu:latest -- /bin/bash
# Inside pod: docker pull <image-name>
```

### Problem: Pod "Ready" but /health returns unhealthy status

**Symptoms**: Pod shows "Running" but `/health` endpoint returns error

**Causes**:
- Model still loading
- Repository ingestion in progress
- Model quantization failed

**Solutions**:
```bash
# Check if pod is still initializing
kubectl describe pod -n rag-api -l app=rag-api | grep -A 20 "Conditions:"

# Check readiness probe
kubectl get pod -n rag-api -l app=rag-api -o yaml | grep -A 10 "readinessProbe:"

# Increase startup probe timeout in Helm
helm upgrade rag-api deploy/helm/rag-api/ \
  -n rag-api \
  --set probes.startup.failureThreshold=120  # 120 × 15s = 30 min

# Monitor progress
kubectl logs -n rag-api -l app=rag-api -f
```

### Problem: ArgoCD shows "OutOfSync"

**Symptoms**: ArgoCD dashboard shows application as "OutOfSync"

**Causes**:
- Git repository changed after deployment
- Kubernetes cluster drifted from desired state
- Image tag changed

**Solutions**:
```bash
# Force ArgoCD sync
argocd app sync rag-api -n argocd

# Or via kubectl
kubectl patch application rag-api -n argocd \
  --type merge \
  -p '{"spec":{"syncPolicy":{"syncOptions":[{"name":"Force=true"}]}}}'

# Check ArgoCD UI for detailed diff
# kubectl port-forward svc/argocd-server -n argocd 8080:443
```

### Problem: Query latency too high

**Symptoms**: `/ask` endpoint takes > 30 seconds

**Causes**:
- Model inference slow (CPU-bound)
- Many embeddings to search
- Insufficient CPU cores

**Solutions**:
```bash
# Profile query execution
curl -v -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Increase CPU allocation
helm upgrade rag-api deploy/helm/rag-api/ \
  -n rag-api \
  --set resources.requests.cpu=1000m \
  --set resources.limits.cpu=4000m

# Check CPU usage
kubectl top pods -n rag-api --containers

# Consider GPU support for 5-10x speedup
# (not recommended for this CPU-optimized model)
```

### Problem: PVC storage full

**Symptoms**: Pod crashes with "disk full" or "no space left on device"

**Causes**:
- Large repository (1GB+ code)
- Many vector embeddings
- Kubernetes logs accumulated

**Solutions**:
```bash
# Check PVC usage
kubectl exec -n rag-api -it deployment/rag-api -- df -h /app/data

# Expand PVC (if storage class supports)
kubectl patch pvc rag-api-pvc -n rag-api \
  --type merge \
  -p '{"spec":{"resources":{"requests":{"storage":"30Gi"}}}}'

# Or delete and recreate with larger size
kubectl delete pvc rag-api-pvc -n rag-api
# Then reapply Helm chart with larger size
```

### Health Probes Not Working

**Symptoms**: Kubernetes reports `CrashLoopBackOff` during startup

**Expected Behavior**:
- **Startup probe**: Waits up to 20 minutes for model to load
- **Readiness probe**: Checks if ready for traffic (ingestion + model loaded)
- **Liveness probe**: Ensures pod stays responsive

**Configuration**:
```yaml
# In deploy/helm/rag-api/values.yaml
probes:
  startup:
    enabled: true
    path: /readyz
    periodSeconds: 15
    timeoutSeconds: 5
    failureThreshold: 80        # 80 × 15s = 1200s (~20 min)
  
  readiness:
    enabled: true
    path: /readyz
    initialDelaySeconds: 20
    periodSeconds: 20
    timeoutSeconds: 5
    failureThreshold: 30
  
  liveness:
    enabled: true
    path: /livez
    initialDelaySeconds: 10
    periodSeconds: 15
    timeoutSeconds: 3
    failureThreshold: 10
```

---

### Kubernetes with Helm

A chart is available at `deploy/helm/rag-api`.

Install:

```bash
helm upgrade --install rag-api deploy/helm/rag-api \
  --set image.repository=ghcr.io/kanha7jpg/github-q-a \
  --set image.tag=runtime
```

Override key values:

```bash
helm upgrade --install rag-api deploy/helm/rag-api \
  --set replicaCount=2 \
  --set resources.requests.memory=4Gi \
  --set resources.limits.memory=6Gi \
  --set persistence.enabled=true \
  --set persistence.size=20Gi
```

The chart includes:

- `replicaCount`
- `resources` with default request memory set to `4Gi`
- `persistence` PVC for Chroma vector DB (`/app/data/chroma`)
- `livenessProbe`, `readinessProbe`, and `startupProbe` tuned for long model-loading windows

### CI/CD (GitOps)

GitLab CI pipeline is defined in `.gitlab-ci.yml` and runs on every push:

- Helm lint + render for `deploy/helm/rag-api`
- Build and push runtime image to GitLab Registry:
  - `$CI_REGISTRY_IMAGE/runtime:$CI_COMMIT_SHA`
  - `$CI_REGISTRY_IMAGE/runtime:$CI_COMMIT_REF_SLUG-latest`
- Build and push quantizer image to GitLab Registry:
  - `$CI_REGISTRY_IMAGE/quantizer:$CI_COMMIT_SHA`
  - `$CI_REGISTRY_IMAGE/quantizer:$CI_COMMIT_REF_SLUG-latest`

ArgoCD manifests are in `deploy/argocd/`:

- `install-minikube.md`: install ArgoCD in Minikube and access UI
- `rag-api-application.yaml`: ArgoCD `Application` that syncs Helm chart from Git

The Application manifest enables Auto-Sync with:

- `prune: true`
- `selfHeal: true`
- `CreateNamespace=true`

Before applying, update placeholders in `deploy/argocd/rag-api-application.yaml`:

- `repoURL`
- `image.repository`
- `targetRevision` if needed

On startup, the app will:

1. Clone or update the target repository.
2. Parse and chunk `.py` and `.md` files.
3. Generate `logs/pipeline-inspection/` when `REPO_URL` changes.
4. Embed chunks and persist them in the local vector store.
5. Load Phi-3.5-mini-instruct for inference.

## API Usage

### Health

```bash
curl http://localhost:8000/health
```

The response reports model loading status via the `status` field, along with `model_loaded`, `ingestion_completed`, and any startup error.

### Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How is request validation implemented?"}'
```

Response includes:

- `answer`: generated response grounded in retrieved snippets
- `sources`: unique source file paths
- `snippets`: top retrieved snippets used as context

### Chunks

```bash
curl "http://localhost:8000/chunks?limit=5&offset=0"
```

Optional query params:

- `limit`: number of chunks to return, capped at 100
- `offset`: pagination offset
- `source`: exact repo-relative file path, such as `src/app.py`

## Notes

- First startup can be slow due to model download and indexing.
- Pipeline inspection files are refreshed automatically the first time you start the app with a new `REPO_URL`.
- If the local Phi model cannot be loaded, the API starts in fallback mode and returns extractive answers from retrieved snippets.
- Startup logs include the generator source and exact local model load error in `logs/process-events.log`.
- CPU inference speed depends on your hardware and model cache state.
- Process completion events are printed to the server console and also written to `logs/process-events.log`.
- To produce GGUF quickly from your local HF snapshot, run `python quantize.py`.
