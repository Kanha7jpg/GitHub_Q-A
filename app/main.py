from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.ingestion import ChunkRecord, RepositoryIngestor
from app.model_bootstrap import ensure_quantized_model
from app.rag import (
    ChromaVectorStore,
    EmbeddingService,
    GeneratorService,
    RagEngine,
    configure_hf_environment,
)
from app.schemas import (
    AskRequest,
    AskResponse,
    ChunkPreview,
    ChunksResponse,
    HealthResponse,
    RetrievedSnippet,
)
from scripts.inspect_pipeline import generate_pipeline_inspection, inspection_needed


@dataclass(slots=True)
class AppState:
    model_loaded: bool = False
    ingestion_completed: bool = False
    error: str | None = None
    rag_engine: RagEngine | None = None
    chunks: list[ChunkRecord] | None = None


PROCESS_LOG_PATH = Path("logs/process-events.log")
INDEX_STATE_PATH = Path("logs/index-state.json")
ANSWER_OUTPUT_DIR = Path("outputs/answers")


def log_process_event(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    PROCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESS_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{line}\n")


def build_index_fingerprint(chunks: list[ChunkRecord]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_index_state() -> dict[str, object] | None:
    if not INDEX_STATE_PATH.exists():
        return None

    try:
        return json.loads(INDEX_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_index_state(payload: dict[str, object]) -> None:
    INDEX_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_answer_output(payload: dict[str, object]) -> Path:
    ANSWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = ANSWER_OUTPUT_DIR / f"ask-{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def should_reindex(chunks: list[ChunkRecord]) -> tuple[bool, str]:
    current_state = {
        "repo_url": settings.repo_url,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model_name": settings.embedding_model_name,
        "collection_name": settings.collection_name,
        "chunk_count": len(chunks),
        "fingerprint": build_index_fingerprint(chunks),
    }
    saved_state = load_index_state()

    if saved_state != current_state:
        save_index_state(current_state)
        if saved_state is None:
            return True, "no prior index state"
        return True, "repository or indexing settings changed"

    return False, "repository unchanged"


settings: Settings = get_settings()
state = AppState()
app = FastAPI(title="Local Repository RAG API", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    global state
    startup_started = perf_counter()
    log_process_event("startup: started")

    try:
        init_started = perf_counter()
        configure_hf_environment(settings)
        ensure_quantized_model(settings, log_process_event)
        ingestor = RepositoryIngestor(settings)
        embedder = EmbeddingService(settings.embedding_model_name)
        vector_store = ChromaVectorStore(settings)
        generator = GeneratorService(settings)
        engine = RagEngine(settings, vector_store, embedder, generator)
        log_process_event(
            f"startup: services initialized in {perf_counter() - init_started:.2f}s"
        )
        log_process_event(f"startup: generator source={generator.model_source}")

        chunk_started = perf_counter()
        chunks = ingestor.build_chunk_records()
        log_process_event(
            f"startup: chunk build finished ({len(chunks)} chunks) in {perf_counter() - chunk_started:.2f}s"
        )

        if inspection_needed(settings.repo_url):
            inspect_started = perf_counter()
            log_process_event("startup: pipeline inspection started")
            generate_pipeline_inspection(settings=settings, emit=log_process_event)
            log_process_event(
                "startup: pipeline inspection finished "
                f"in {perf_counter() - inspect_started:.2f}s"
            )
        else:
            log_process_event("startup: pipeline inspection skipped (repo_url unchanged)")

        reindex_required, reindex_reason = should_reindex(chunks)
        if reindex_required:
            index_started = perf_counter()
            log_process_event(f"startup: vector indexing started ({reindex_reason})")
            engine.index_chunks(chunks)
            log_process_event(
                f"startup: vector indexing finished in {perf_counter() - index_started:.2f}s"
            )
        else:
            log_process_event(
                f"startup: vector indexing skipped ({reindex_reason})"
            )

        state.rag_engine = engine
        state.chunks = chunks
        state.model_loaded = generator.model is not None
        state.ingestion_completed = True
        state.error = generator.load_error

        if state.model_loaded:
            log_process_event("startup: generator model loaded")
        else:
            log_process_event("startup: generator fallback mode active")
            if state.error:
                log_process_event(f"startup: generator load error={state.error}")

        log_process_event(
            f"startup: completed successfully in {perf_counter() - startup_started:.2f}s"
        )
    except Exception as exc:  # noqa: BLE001
        state.error = str(exc)
        state.model_loaded = False
        state.ingestion_completed = False
        log_process_event(f"startup: failed - {exc}")
        log_process_event(f"startup: traceback\n{traceback.format_exc()}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if state.model_loaded:
        status = "loaded"
    elif state.error:
        status = "error"
    else:
        status = "loading"

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
        "chunks": "/chunks",
        "ask": "/ask",
    }


@app.get("/chunks", response_model=ChunksResponse)
def chunks(
    limit: int = 20,
    offset: int = 0,
    source: str | None = None,
) -> ChunksResponse:
    if state.chunks is None:
        raise HTTPException(status_code=503, detail="Chunks not ready")

    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)

    filtered_chunks = state.chunks
    if source:
        filtered_chunks = [chunk for chunk in filtered_chunks if chunk.source == source]

    window = filtered_chunks[safe_offset : safe_offset + safe_limit]
    previews = [
        ChunkPreview(chunk_id=chunk.chunk_id, source=chunk.source, text=chunk.text)
        for chunk in window
    ]

    return ChunksResponse(
        total=len(filtered_chunks),
        offset=safe_offset,
        limit=safe_limit,
        source=source,
        chunks=previews,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if state.rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    if settings.slm_backend == "llama_cpp" and state.rag_engine.generator.backend != "llama_cpp":
        raise HTTPException(
            status_code=503,
            detail=(
                "Quantized model backend is required but not loaded. "
                f"Error: {state.rag_engine.generator.load_error or 'unknown'}"
            ),
        )

    request_started = perf_counter()
    log_process_event("ask: request started")
    answer, snippets = await run_in_threadpool(state.rag_engine.answer_question, request.query)
    log_process_event(
        f"ask: retrieval and generation completed in {perf_counter() - request_started:.2f}s"
    )

    unique_sources = sorted({snippet.source for snippet in snippets})
    response_snippets = [
        RetrievedSnippet(source=snippet.source, text=snippet.text) for snippet in snippets
    ]

    log_process_event(
        f"ask: response ready with {len(response_snippets)} snippets from {len(unique_sources)} sources"
    )

    try:
        output_path = save_answer_output(
            {
                "query": request.query,
                "answer": answer,
                "sources": unique_sources,
                "snippets": [
                    {"source": snippet.source, "text": snippet.text}
                    for snippet in response_snippets
                ],
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        log_process_event(f"ask: saved answer output to {output_path.as_posix()}")
    except Exception as exc:  # noqa: BLE001
        log_process_event(f"ask: failed to save answer output - {exc}")

    return AskResponse(answer=answer, sources=unique_sources, snippets=response_snippets)
