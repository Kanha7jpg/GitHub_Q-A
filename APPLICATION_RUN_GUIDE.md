# Application Run Guide

This document explains how the Local Repository RAG API runs, how requests flow through the system, and how to run and verify it manually.

## 1. What the Application Does

The API answers repository questions using Retrieval-Augmented Generation (RAG):

1. Ingest repository files (.py and .md).
2. Chunk and embed content into Chroma.
3. Retrieve top 3 relevant snippets for each question.
4. Generate an answer from retrieved context.
5. Save each answer to outputs/answers as JSON.

## 2. Runtime Architecture

Core components:

1. RepositoryIngestor:
- Clones/pulls repository content.
- Filters .py and .md files.
- Produces chunk records.

2. EmbeddingService:
- Encodes chunks and queries for retrieval.

3. ChromaVectorStore:
- Persistent vector storage under data/chroma.

4. GeneratorService:
- Uses llama.cpp GGUF model when configured (quantized mode).
- Falls back to Transformers only if backend configuration allows it.

5. RagEngine:
- Retrieves REQUIRED_TOP_K = 3 snippets.
- Builds context string and generates answer.

## 3. Startup Flow

When uvicorn starts app.main:app:

1. Load settings from .env.
2. Configure HF cache environment variables.
3. Initialize ingestor, embedding service, vector store, generator, and RAG engine.
4. Build chunk records from repository.
5. Optionally run pipeline inspection if repository URL changed.
6. Reindex vectors if repository/indexing fingerprint changed.
7. Mark startup status and log results to logs/process-events.log.

## 4. Ask Request Flow

For POST /ask:

1. Validate RAG engine readiness.
2. If strict quantized backend is required, ensure llama_cpp backend is active.
3. Embed user query.
4. Retrieve top 3 snippets from Chroma.
5. Build prompt:
   Context: {retrieved_snippets}
   Question: {user_query}
6. Generate concise answer from model.
7. Save answer JSON to outputs/answers/ask-<timestamp>.json.
8. Return answer, sources, and snippets in API response.

## 5. Backend Selection (Quantized vs Transformers)

Relevant .env keys:

- SLM_BACKEND
  - llama_cpp: require quantized GGUF backend only.
  - auto: prefer GGUF when available, otherwise may use Transformers.
  - transformers: force HF Transformers backend.

- GGUF_MODEL_PATH
  - Path to quantized .gguf file.

Current strict quantized setup should use:

- SLM_BACKEND=llama_cpp
- GGUF_MODEL_PATH=<absolute path to phi-3.5-q4_k_m.gguf>

## 6. Folders Created/Used During Runtime

1. data/repository
- Cloned/pulled repository content.

2. data/chroma
- Persistent Chroma vector DB storage.

3. logs
- process-events.log and index-state.json.

4. logs/pipeline-inspection
- Inspection summaries and sample artifacts.

5. outputs/answers
- Saved answer payloads from /ask.

6. .cache/hf_modules
- Hugging Face modules cache used during model loading.

7. data/models
- Quantized GGUF output files from quantize.py.

## 7. Manual Run Commands (PowerShell)

From project root:

1. Activate environment:

   .\.venv\Scripts\Activate.ps1

2. Start API:

   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

3. Check health (new terminal):

   Invoke-RestMethod -Uri http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5

4. Ask question:

   $body = @{ query = 'What is this repository about?' } | ConvertTo-Json
   Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/ask' -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 8

5. Check latest saved output:

   Get-ChildItem outputs/answers -File | Sort-Object LastWriteTime -Descending | Select-Object -First 3 FullName, LastWriteTime, Length

6. Check runtime logs:

   Get-Content logs/process-events.log -Tail 40

7. Stop server:

   Ctrl+C in the server terminal.

## 8. Quantization Workflow

1. Ensure CMake and C/C++ build tools are installed.
2. Run quantization:

   python quantize.py

3. Script outputs GGUF model in data/models and updates .env with GGUF_MODEL_PATH.

## 9. Common Checks

1. Confirm quantized model is active:
- Look for generator source ending in .gguf in logs/process-events.log.

2. Confirm strict quantized-only behavior:
- Keep SLM_BACKEND=llama_cpp in .env.
- If GGUF cannot load, /ask should return 503 instead of using non-quantized answers.

3. Confirm answer persistence:
- Each successful /ask appends one JSON file in outputs/answers.

## 10. Quick Troubleshooting

1. Error: cmake not found
- Install CMake, then reopen terminal.

2. Error: C compiler not found (nmake/cl missing)
- Install Visual Studio Build Tools 2022 with C++ workload.

3. Error: llama_cpp import missing
- Install dependency in active venv:

  pip install llama-cpp-python

4. Slow first startup
- Expected when indexing or loading model for first time.
- Later runs are usually faster when index is unchanged.
