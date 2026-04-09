from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
import re
from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import Settings
from app.ingestion import ChunkRecord


@dataclass(slots=True)
class SearchResult:
    source: str
    text: str


class EmbeddingService:
    """Encodes text into vectors using a local sentence transformer model."""

    def __init__(self, model_name: str) -> None:
        self.fallback_dimension = 384
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

    def _fallback_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.fallback_dimension
        tokens = re.findall(r"\w+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.fallback_dimension
            vector[index] += 1.0

        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            vector = [value / magnitude for value in vector]

        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            return [self._fallback_embed(text) for text in texts]

        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        if self.model is None:
            return self._fallback_embed(query)

        return self.model.encode(query, normalize_embeddings=True).tolist()


class ChromaVectorStore:
    """Persistent local vector storage wrapper."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.settings.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
            
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": chunk.source} for chunk in chunks]
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        search_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            for doc, meta in zip(docs, metas, strict=False):
                search_results.append(SearchResult(source=meta["source"], text=doc))
                
        return search_results


class GeneratorService:
    """Loads Phi-3.5-mini-instruct locally and generates grounded answers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.slm_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.slm_model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation=settings.slm_attn_implementation,
            )
            self.load_error = None
        except Exception as e:
            self.tokenizer = None
            self.model = None
            self.load_error = str(e)

    def _fallback_generate(self, context: str) -> str:
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        relevant_lines = [line for line in lines if not line.startswith("[Source:")]

        if not relevant_lines:
            return "I could not find enough repository context to answer this question."

        excerpt = " ".join(relevant_lines[:3])
        if len(excerpt) > 500:
            excerpt = f"{excerpt[:497]}..."

        return f"Model is not cached locally yet. Relevant repository context: {excerpt}"

    def generate(self, context: str, question: str) -> str:
        if self.model is None or self.tokenizer is None:
            return self._fallback_generate(context)

        user_prompt = f"Context: {context}\nQuestion: {question}"
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer technical questions using only the provided context. "
                    "If context is insufficient, explicitly say that. "
                    "Keep the answer concise."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs,
                    max_new_tokens=self.settings.max_new_tokens,
                    do_sample=self.settings.temperature > 0,
                    temperature=self.settings.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # Work around DynamicCache API mismatches in some
                    # transformers + Phi-3 remote-code combinations.
                    use_cache=False,
                )
        except Exception:
            return self._fallback_generate(context)

        generated_ids = output_ids[0][inputs.shape[-1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer


class RagEngine:
    """Coordinates retrieval and generation for question answering."""

    def __init__(
        self,
        settings: Settings,
        vector_store: ChromaVectorStore,
        embedder: EmbeddingService,
        generator: GeneratorService,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.embedder = embedder
        self.generator = generator

    def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        self.vector_store.reset_collection()
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.vector_store.upsert_chunks(chunks, embeddings)

    def answer_question(self, query: str) -> tuple[str, list[SearchResult]]:
        query_embedding = self.embedder.embed_query(query)
        snippets = self.vector_store.query(query_embedding, top_k=self.settings.top_k)

        if not snippets:
            return "I could not find relevant repository context for this question.", []

        context = "\n\n".join(
            f"[Source: {snippet.source}]\n{snippet.text}"
            for snippet in snippets
        )
        answer = self.generator.generate(context=context, question=query)
        return answer, snippets
