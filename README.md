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
