from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.ingestion import RepositoryIngestor
from app.rag import ChromaVectorStore, EmbeddingService, GeneratorService, RagEngine
from app.schemas import AskRequest, AskResponse, HealthResponse, RetrievedSnippet


@dataclass(slots=True)
class AppState:
    model_loaded: bool = False
    ingestion_completed: bool = False
    error: str | None = None
    rag_engine: RagEngine | None = None


settings: Settings = get_settings()
state = AppState()
app = FastAPI(title="Local Repository RAG API", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    global state

    try:
        ingestor = RepositoryIngestor(settings)
        embedder = EmbeddingService(settings.embedding_model_name)
        vector_store = ChromaVectorStore(settings)
        generator = GeneratorService(settings)
        engine = RagEngine(settings, vector_store, embedder, generator)

        chunks = ingestor.build_chunk_records()
        engine.index_chunks(chunks)

        state.rag_engine = engine
        state.model_loaded = generator.model is not None
        state.ingestion_completed = True
        state.error = generator.load_error
    except Exception as exc:  # noqa: BLE001
        state.error = str(exc)
        state.model_loaded = False
        state.ingestion_completed = False


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = "ok" if state.model_loaded and state.ingestion_completed else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=state.model_loaded,
        ingestion_completed=state.ingestion_completed,
        error=state.error,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Local Repository RAG API",
        "health": "/health",
        "ask": "/ask",
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if state.rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    answer, snippets = await run_in_threadpool(state.rag_engine.answer_question, request.query)

    unique_sources = sorted({snippet.source for snippet in snippets})
    response_snippets = [
        RetrievedSnippet(source=snippet.source, text=snippet.text) for snippet in snippets
    ]

    return AskResponse(answer=answer, sources=unique_sources, snippets=response_snippets)
