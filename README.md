# Local Repository RAG (FastAPI + Phi-3.5)

This project builds a local Retrieval-Augmented Generation (RAG) API that answers technical questions about a repository cloned from a URL in environment variables.

## Features

- Clones a public repository from `REPO_URL`.
- Ingests only `.py` and `.md` files.
- Uses recursive chunking with Python-aware splitting for source files.
- Builds embeddings with a local Hugging Face sentence-transformer model.
- Stores vectors in a persistent local store (`data/chroma`).
- Retrieves top 3 snippets and prompts `microsoft/Phi-3.5-mini-instruct`.
- Exposes:
  - `GET /health`
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
```

You can also point `REPO_URL` at a local directory to run the app offline against files in that folder.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On startup, the app will:

1. Clone or update the target repository.
2. Parse and chunk `.py` and `.md` files.
3. Embed chunks and persist them in the local vector store.
4. Load Phi-3.5-mini-instruct for inference.

## API Usage

### Health

```bash
curl http://localhost:8000/health
```

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

## Notes

- First startup can be slow due to model download and indexing.
- If the Hugging Face model is not already cached locally, the API starts in fallback mode and returns extractive answers from retrieved snippets.
- CPU inference speed depends on your hardware and model cache state.
