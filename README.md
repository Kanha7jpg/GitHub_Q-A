# Local Repository RAG (FastAPI + Phi-3.5)

This project builds a local Retrieval-Augmented Generation (RAG) API that answers technical questions about a repository cloned from a URL in environment variables.

Detailed runtime documentation: see `APPLICATION_RUN_GUIDE.md`.

## Features

- Clones a public repository from `REPO_URL`.
- Ingests only `.py` and `.md` files.
- Uses recursive chunking with Python-aware splitting for source files.
- Builds embeddings with a local Hugging Face sentence-transformer model.
- Stores vectors in a persistent local store (`data/chroma`).
- Retrieves top 3 snippets and prompts `microsoft/Phi-3.5-mini-instruct`.
- Exposes:
  - `GET /health`
  - `GET /chunks`
  - `POST /ask`

## Requirements

- Python 3.10+
- `git` installed and available in `PATH`

## Setup

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
