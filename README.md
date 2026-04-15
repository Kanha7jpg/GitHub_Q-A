# GitHub Q&A - Local Repository RAG API

A production-grade Retrieval Augmented Generation (RAG) system that ingests GitHub repositories and provides intelligent question-answering capabilities using locally-quantized language models. Built with FastAPI, LangChain, ChromaDB, and Phi-3.5 Mini.

## 🌟 Features

- **Local LLM Inference**: Uses Phi-3.5-mini quantized to Q4_K_M format (~2.3GB) for efficient on-device inference
- **Repository Ingestion**: Automatically clones and processes GitHub repositories with configurable chunking
- **Vector Search**: ChromaDB-based semantic search using sentence transformers
- **Production Ready**: Kubernetes/Helm deployments, Docker containerization, comprehensive health checks
- **Flexible Backend**: Support for Transformers, Llama.cpp, or auto-detection for model inference
- **API-First Design**: RESTful API with comprehensive logging and inspection tools

## 🏗️ Architecture

```
User Request
    ↓
[FastAPI Server]
    ↓
┌─────────────────────────────────┐
│ Request Processing              │
├─────────────────────────────────┤
│ 1. Parse user query             │
│ 2. Embedding Service            │
│    ↓                            │
│    Query Embedding (384-dim)    │
└─────────────────────────────────┘
    ↓
[ChromaDB Vector Database]
    ↓
┌─────────────────────────────────┐
│ Semantic Search (top-k chunks)  │
├─────────────────────────────────┤
│ Retrieved Documents & Metadata  │
└─────────────────────────────────┘
    ↓
[Prompt Construction]
    ↓
[Phi-3.5-mini (Q4_K_M Quantized)]
    ↓
[Generated Answer]
    ↓
[HTTP Response with Source References]
```

### Component Breakdown

```
Data Pipeline (Ingestion):
┌──────────────────┐
│ GitHub Repo URL  │
└────────┬─────────┘
         ↓
┌──────────────────────────────────┐
│ Clone & Extract Files            │
│ (Repository ingestion.py)        │
└────────┬─────────────────────────┘
         ↓
┌──────────────────────────────────┐
│ Text Chunking                    │
│ chunk_size: 900, overlap: 150    │
└────────┬─────────────────────────┘
         ↓
┌──────────────────────────────────┐
│ Embedding Generation             │
│ (all-MiniLM-L6-v2)               │
└────────┬─────────────────────────┘
         ↓
┌──────────────────────────────────┐
│ Store in ChromaDB                │
│ (/data/chroma/collection)        │
└──────────────────────────────────┘

Inference Pipeline:
Query ──→ [Embedding] ──→ [Vector Search] ──→ [Top-K Retrieval]
                                                      ↓
                                            [Context Window]
                                                      ↓
                                            [Phi-3.5 Inference]
                                                      ↓
                                            [Formatted Response]
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 6GB RAM (for model inference)
- 20GB disk space (for models, vectors, and repository)
- CUDA-capable GPU (optional but recommended)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kanha7jpg/GitHub_Q-A.git
   cd GitHub_Q-A
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Quantize the model** (one-time setup)
   ```bash
   python quantize.py \
     --output-dir models/phi3_quantized \
     --quantized-name phi-3.5-q4_k_m-local.gguf \
     --backend llama_cpp
   ```
   This creates a ~2.3GB quantized model in the `models/phi3_quantized/` directory.

5. **Create environment configuration**
   ```bash
   cat > .env << EOF
   REPO_URL=https://github.com/<user>/<repo-name>
   CLONE_DIR=data/repository
   CHROMA_PATH=data/chroma
   COLLECTION_NAME=repo_chunks
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   SLM_MODEL_NAME=microsoft/Phi-3.5-mini-instruct
   SLM_BACKEND=llama_cpp
   GGUF_MODEL_PATH=models/phi3_quantized/phi-3.5-q4_k_m-local.gguf
   SLM_LOCAL_FILES_ONLY=true
   LLAMA_N_THREADS=8
   CHUNK_SIZE=900
   CHUNK_OVERLAP=150
   TOP_K=3
   MAX_NEW_TOKENS=50
   TEMPERATURE=0.2
   EOF
   ```

6. **Start the API server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the API**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - OpenAPI Schema: http://localhost:8000/openapi.json

## 🐳 Docker Deployment

### Building Docker Images

```bash
# Build runtime image (with quantized model embedded)
docker build \
  --target runtime \
  --build-arg MODEL_NAME=microsoft/Phi-3.5-mini-instruct \
  -t ghcr.io/kanha7jpg/github-q-a/runtime:latest \
  .

# Build quantizer image (for Kubernetes init container)
docker build \
  --target quantizer \
  -t ghcr.io/kanha7jpg/github-q-a/quantizer:latest \
  .
```

### Running with Docker

```bash
docker run -d \
  -p 8000:8000 \
  -v models:/opt/models \
  -v chroma:/app/data/chroma \
  -e REPO_URL=https://github.com/Kanha7jpg/BasicLearning \
  -e SLM_BACKEND=llama_cpp \
  --name rag-api \
  ghcr.io/kanha7jpg/github-q-a/runtime:latest
```

## ☸️ Kubernetes Deployment with Helm

### Prerequisites

- Kubernetes cluster (1.24+)
- Helm 3.x
- kubectl configured with cluster access
- 6GB available memory per node
- 20GB storage provisioner

### Installation Steps

1. **Update values for your environment**
   ```bash
   cd deploy/helm/rag-api
   
   # Edit values.yaml with your settings:
   # - Repository URL to ingest
   # - Storage class and size
   # - Resource requests/limits
   # - Image registry credentials (if private)
   ```

2. **Deploy using Helm**
   ```bash
   # Add any required image pull secrets
   kubectl create secret docker-registry ghcr-secret \
     --docker-server=ghcr.io \
     --docker-username=<username> \
     --docker-password=<token>

   # Install or upgrade the release
   helm upgrade --install rag-api ./deploy/helm/rag-api \
     --namespace rag-system \
     --create-namespace \
     --values deploy/helm/rag-api/values.yaml
   ```

3. **Verify deployment**
   ```bash
   kubectl get pods -n rag-system
   kubectl logs -n rag-system -l app=rag-api -f
   ```

4. **Access the API**
   ```bash
   # Port-forward for local access
   kubectl port-forward -n rag-system svc/rag-api 8000:8000

   # Or configure Ingress (see templates/ingress.yaml)
   ```

### Helm Configuration Options

Key values in `values.yaml`:

```yaml
# Application
replicaCount: 1
image:
  repository: ghcr.io/kanha7jpg/github-q-a/runtime
  tag: latest

# Storage
persistence:
  enabled: true
  size: 20Gi
  storageClassName: default  # Change as needed

# Model initialization
modelInit:
  enabled: true  # Downloads and quantizes model on first run

# Resources
resources:
  requests:
    cpu: "500m"
    memory: 4Gi
  limits:
    cpu: "2"
    memory: 6Gi

# Environment
env:
  REPO_URL: https://github.com/Kanha7jpg/BasicLearning
  SLM_BACKEND: llama_cpp
  LLAMA_N_THREADS: "8"
```

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ingestion_completed": true
}
```

### Ask Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do you implement a binary search?"}'
```

Response:
```json
{
  "query": "How do you implement a binary search?",
  "answer": "Binary search is implemented by...",
  "retrieved_chunks": [
    {
      "source": "Python/algorithms.py",
      "text": "def binary_search(arr, target)..."
    }
  ],
  "processing_time_ms": 245
}
```

### Get Repository Chunks
```bash
curl http://localhost:8000/chunks?limit=10&offset=0
```

## ⚙️ Configuration

All configuration is via environment variables loaded from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO_URL` | Required | GitHub repository URL to ingest |
| `SLM_BACKEND` | `auto` | Model backend: `transformers`, `llama_cpp`, or `auto` |
| `GGUF_MODEL_PATH` | None | Path to quantized GGUF model |
| `SLM_LOCAL_FILES_ONLY` | `true` | Use only cached HF models (offline mode) |
| `LLAMA_N_THREADS` | Auto | Thread count for Llama.cpp inference |
| `LLAMA_N_CTX` | `2048` | Context window size |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between chunks |
| `TOP_K` | `3` | Retrieved chunks per query |
| `MAX_NEW_TOKENS` | `50` | Max tokens in generated answer |
| `TEMPERATURE` | `0.2` | Sampling temperature (lower = deterministic) |

## 🔧 Troubleshooting

### Local Setup

**Issue**: Import errors for `llama_cpp_python`
```
ModuleNotFoundError: No module named 'llama_cpp'
```
**Solution**: 
- Ensure CMake is installed: `cmake --version`
- On Windows without Visual Studio: `pip install cmake`
- Rebuild: `pip install --force-reinstall llama-cpp-python`

**Issue**: Model not loading, "GGUF file not found"
```
FileNotFoundError: models/phi3_quantized/phi-3.5-q4_k_m-local.gguf
```
**Solution**:
- Run quantization: `python quantize.py --backend llama_cpp`
- Verify file exists: `ls -la models/phi3_quantized/`
- Check `GGUF_MODEL_PATH` in `.env` matches actual location

**Issue**: Out of memory during quantization
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Use CPU backend: `--backend transformers`
- Reduce batch size in `quantize.py`
- Increase system swap space

### Kubernetes / Helm Troubleshooting

#### Pod Stuck in Pending

```bash
kubectl describe pod -n rag-system <pod-name>
```

**Common causes and solutions**:

1. **Insufficient resources**
   ```bash
   # Check node capacity
   kubectl top nodes
   kubectl describe node <node-name>
   
   # Reduce requested resources in values.yaml
   resources:
     requests:
       memory: 2Gi  # Reduced from 4Gi
   ```

2. **PVC not provisioned**
   ```bash
   kubectl get pvc -n rag-system
   kubectl describe pvc -n rag-system <pvc-name>
   
   # Verify storage class exists
   kubectl get storageclass
   
   # If missing, create one:
   kubectl apply -f deploy/helm/rag-api/templates/storageclass.yaml
   ```

3. **ImagePullBackOff** (Cannot pull container image)
   ```bash
   kubectl describe pod -n rag-system <pod-name> | grep -A 10 "Events:"
   
   # Solutions:
   # a) Create image pull secret
   kubectl create secret docker-registry ghcr-secret \
     --docker-server=ghcr.io \
     --docker-username=<user> \
     --docker-password=<token> \
     -n rag-system
   
   # b) Update values.yaml with imagePullSecrets
   imagePullSecrets:
     - name: ghcr-secret
   
   # c) Use public image tag
   helm upgrade rag-api ./deploy/helm/rag-api \
     --set image.tag=latest \
     -n rag-system
   ```

#### Pod CrashLoopBackOff

```bash
# Check logs
kubectl logs -n rag-system <pod-name> --tail=50
kubectl logs -n rag-system <pod-name> --previous  # Previous crash logs
```

**Common causes**:

1. **Model download fails**
   ```
   Error: Failed to download Phi-3.5-mini from HuggingFace
   ```
   - Verify internet connectivity: `kubectl exec -it <pod> -- ping huggingface.co`
   - Check HF_TOKEN if using gated models
   - Increase model init timeout in values.yaml

2. **Storage permission denied**
   ```
   PermissionError: /app/data/chroma: Permission denied
   ```
   ```bash
   # Fix in values.yaml:
   podSecurityContext:
     fsGroup: 1000
   ```

3. **Port already in use**
   ```
   Address already in use: ('0.0.0.0', 8000)
   ```
   - Multiple replicas on same node (normal in cluster)
   - If local testing: `lsof -i :8000` and kill process

#### Health Check Failures

```bash
# Monitor health checks
kubectl get event -n rag-system --sort-by='.lastTimestamp'

# Check detailed pod status
kubectl describe pod -n rag-system <pod-name>
```

**Health check issues**:

1. **Model initialization timeout**
   - Default: 5 minutes
   - Increase in values.yaml:
   ```yaml
   livenessProbe:
     initialDelaySeconds: 600  # 10 minutes
     timeoutSeconds: 30
   ```

2. **Repository ingestion still processing**
   ```bash
   # Check pod logs during initialization
   kubectl logs -n rag-system -f <pod-name>
   
   # Once ingestion completes, health will return 200
   ```

#### Persistent Volume (PV) Issues

```bash
# Check PVC status
kubectl get pvc -n rag-system
kubectl describe pvc -n rag-system repo-chunks-pvc

# If stuck in Pending, check storage provisioner
kubectl get storageclass
kubectl describe storageclass <class-name>

# For local testing with Minikube:
minikube addons enable default-storageclass
minikube mount $(pwd)/data:/data --uid=1000 --gid=1000
```

#### Model Loading Failures in Llama.cpp Backend

```bash
# Check initialization container logs
kubectl logs -n rag-system <pod-name> -c model-init

# Common issues:
# - GGUF file corrupted: Re-download
# - Incompatible CUDA version: Use CPU backend
# - Insufficient memory: Reduce LLAMA_N_THREADS
```

**Solution**:
```bash
# Update values.yaml
env:
  SLM_BACKEND: transformers  # Fallback to transformers
  LLAMA_N_THREADS: "4"       # Reduce threads
```

#### Memory and CPU Limits

```bash
# Monitor resource usage
kubectl top pod -n rag-system

# If exceeding limits, adjust in values.yaml:
resources:
  requests:
    memory: 4Gi
  limits:
    memory: 8Gi  # Increased headroom
    
# Verify settings applied
kubectl get pod -n rag-system <pod-name> -o yaml | grep -A 5 "resources:"
```

#### Debugging with kubectl Exec

```bash
# Get shell access to pod
kubectl exec -it -n rag-system <pod-name> -- /bin/bash

# Inside pod:
# - Check logs: tail -f logs/process-events.log
# - Test API: curl http://localhost:8000/health
# - Check model: ls -la models/phi3_quantized/
# - Verify ChromaDB: ls -la data/chroma/
# - Check environment: env | grep -E "(REPO|MODEL|BACKEND)"
```

### Performance Optimization

**Slow query responses**:
1. Reduce `TOP_K` in config (fewer retrieved chunks)
2. Increase `MAX_NEW_TOKENS` limit for Llama.cpp
3. Use GPU acceleration (set CUDA_VISIBLE_DEVICES)
4. Scale replicas: `kubectl scale deployment -n rag-system rag-api --replicas=3`

**High memory usage**:
1. Reduce `LLAMA_N_CTX` (context window size)
2. Use `LLAMA_N_THREADS=4` (fewer threads)
3. Verify `SLM_LOCAL_FILES_ONLY=true` to avoid re-downloads

## 📊 Logs and Inspection

```bash
# View real-time logs
tail -f logs/process-events.log

# Inspect pipeline (embeddings, chunks, etc.)
python scripts/inspect_pipeline.py

# Check index state
cat logs/index-state.json | jq .

# View previous Q&A outputs
ls outputs/answers/
cat outputs/answers/ask-*.json | jq .
```

## 🛠️ Development

### Running Tests
```bash
# (Add test configuration as needed)
pytest tests/
```

### Code Structure
```
app/
├── main.py              # FastAPI application and endpoints
├── rag.py              # RAG engine, embedding, and generation services
├── ingestion.py        # Repository ingestion and chunking
├── model_bootstrap.py  # Model quantization and loading
├── config.py           # Configuration management
└── schemas.py          # Request/response schemas

deploy/
├── helm/rag-api/       # Helm chart for K8s deployment
├── k8s/                # Raw K8s manifests
└── argocd/             # ArgoCD deployment configs

scripts/
└── inspect_pipeline.py # Debugging and inspection tools

quantize.py            # Model quantization script
```

## 📝 Environment Variables Detailed

### Model Configuration
- `SLM_MODEL_NAME`: HuggingFace model ID (default: `microsoft/Phi-3.5-mini-instruct`)
- `SLM_BACKEND`: Inference backend (`transformers`, `llama_cpp`, `auto`)
- `GGUF_MODEL_PATH`: Path to quantized GGUF file
- `SLM_LOCAL_FILES_ONLY`: Use only cached models (offline mode)

### Llama.cpp Specific
- `LLAMA_N_CTX`: Context window size (2048 recommended for Phi-3.5)
- `LLAMA_N_THREADS`: CPU threads for inference
- `LLAMA_N_GPU_LAYERS`: Layers offloaded to GPU (0 = CPU only)

### Data Pipeline
- `REPO_URL`: GitHub repository to ingest
- `CHUNK_SIZE`: Characters per text chunk (900 default)
- `CHUNK_OVERLAP`: Character overlap between chunks (150 default)
- `TOP_K`: Retrieved chunks per query (3 default)

### Generation
- `MAX_NEW_TOKENS`: Maximum response length (50 tokens ≈ 150 words)
- `TEMPERATURE`: Sampling temperature (0.2 = deterministic)

## 🔗 References

- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Troubleshooting](https://kubernetes.io/docs/tasks/debug-application-cluster/)

---

**Last Updated**: April 2026  

