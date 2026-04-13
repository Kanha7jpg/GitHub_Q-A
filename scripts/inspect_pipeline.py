from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.ingestion import RepositoryIngestor
from app.rag import EmbeddingService


OUTPUT_DIR = Path("logs/pipeline-inspection")


def write_json(name: str, payload: object) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / name
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_existing_summary() -> dict[str, object] | None:
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        return None

    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def inspection_needed(repo_url: str) -> bool:
    existing_summary = load_existing_summary()
    if existing_summary is None:
        return True

    return existing_summary.get("repo_url") != repo_url


def generate_pipeline_inspection(
    settings=None,
    emit: Callable[[str], None] | None = print,
) -> dict[str, object]:
    settings = settings or get_settings()
    ingestor = RepositoryIngestor(settings)
    logger = emit or (lambda *_args, **_kwargs: None)

    logger("Pipeline inspection")
    logger(f"repo_url={settings.repo_url}")
    logger(f"clone_dir={settings.clone_dir}")
    logger(f"chunk_size={settings.chunk_size}")
    logger(f"chunk_overlap={settings.chunk_overlap}")
    logger("")

    clone_started = perf_counter()
    repo_path = ingestor.clone_repository()
    clone_elapsed = perf_counter() - clone_started
    logger("1. Repository stage")
    logger(f"repo_path={repo_path}")
    logger(f"elapsed={clone_elapsed:.2f}s")
    logger("")

    parse_started = perf_counter()
    files = ingestor._iter_supported_files(repo_path)
    parse_elapsed = perf_counter() - parse_started
    file_samples = [
        str(path.relative_to(repo_path)).replace("\\", "/") for path in files[:10]
    ]
    logger("2. Parse stage")
    logger(f"supported_files={len(files)}")
    logger(f"elapsed={parse_elapsed:.2f}s")
    logger("sample_files:")
    for sample in file_samples:
        logger(f"  - {sample}")
    logger("")

    chunk_started = perf_counter()
    chunk_records = ingestor.build_chunk_records()
    chunk_elapsed = perf_counter() - chunk_started
    chunk_counts = Counter(chunk.source for chunk in chunk_records)
    top_chunked_files = chunk_counts.most_common(10)
    chunk_samples = [
        {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "text_preview": chunk.text[:240],
            "length": len(chunk.text),
        }
        for chunk in chunk_records[:10]
    ]
    logger("3. Chunk stage")
    logger(f"chunks={len(chunk_records)}")
    logger(f"elapsed={chunk_elapsed:.2f}s")
    logger("top_files_by_chunk_count:")
    for source, count in top_chunked_files:
        logger(f"  - {source}: {count}")
    logger("")

    embed_started = perf_counter()
    embedder = EmbeddingService(settings.embedding_model_name)
    embeddings = embedder.embed_documents([chunk.text for chunk in chunk_records])
    embed_elapsed = perf_counter() - embed_started
    embedding_dimension = len(embeddings[0]) if embeddings else 0
    embedding_samples = [
        {
            "chunk_id": chunk_records[index].chunk_id,
            "source": chunk_records[index].source,
            "embedding_dimension": len(embeddings[index]),
            "embedding_preview": embeddings[index][:8],
        }
        for index in range(min(5, len(chunk_records)))
    ]
    logger("4. Embed stage")
    logger(f"embedded_chunks={len(embeddings)}")
    logger(f"embedding_dimension={embedding_dimension}")
    logger(f"model_loaded={embedder.model is not None}")
    logger(f"elapsed={embed_elapsed:.2f}s")

    summary = {
        "repo_url": settings.repo_url,
        "repo_path": str(repo_path),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "stages": {
            "repository": {
                "elapsed_seconds": round(clone_elapsed, 2),
            },
            "parse": {
                "supported_files": len(files),
                "elapsed_seconds": round(parse_elapsed, 2),
                "sample_files": file_samples,
            },
            "chunk": {
                "chunks": len(chunk_records),
                "elapsed_seconds": round(chunk_elapsed, 2),
                "top_files_by_chunk_count": [
                    {"source": source, "count": count} for source, count in top_chunked_files
                ],
            },
            "embed": {
                "embedded_chunks": len(embeddings),
                "embedding_dimension": embedding_dimension,
                "model_loaded": embedder.model is not None,
                "elapsed_seconds": round(embed_elapsed, 2),
            },
        },
    }

    write_json("summary.json", summary)
    write_json("chunk_samples.json", chunk_samples)
    write_json("embedding_samples.json", embedding_samples)

    logger("")
    logger(f"Saved summary to {OUTPUT_DIR / 'summary.json'}")
    logger(f"Saved chunk samples to {OUTPUT_DIR / 'chunk_samples.json'}")
    logger(f"Saved embedding samples to {OUTPUT_DIR / 'embedding_samples.json'}")
    return summary


def main() -> None:
    generate_pipeline_inspection()


if __name__ == "__main__":
    main()
