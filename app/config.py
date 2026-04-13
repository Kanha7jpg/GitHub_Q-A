from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    repo_url: str = Field(alias="REPO_URL")
    clone_dir: Path = Field(default=Path("data/repository"), alias="CLONE_DIR")
    chroma_path: Path = Field(default=Path("data/chroma"), alias="CHROMA_PATH")
    collection_name: str = Field(default="repo_chunks", alias="COLLECTION_NAME")

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    slm_model_name: str = Field(
        default="microsoft/Phi-3.5-mini-instruct",
        alias="SLM_MODEL_NAME",
    )
    slm_backend: Literal["auto", "transformers", "llama_cpp"] = Field(
        default="auto",
        alias="SLM_BACKEND",
    )
    slm_model_path: Path | None = Field(default=None, alias="SLM_MODEL_PATH")
    gguf_model_path: Path | None = Field(default=None, alias="GGUF_MODEL_PATH")
    slm_local_files_only: bool = Field(default=True, alias="SLM_LOCAL_FILES_ONLY")
    slm_modules_cache: Path = Field(
        default=Path(".cache/hf_modules"),
        alias="SLM_MODULES_CACHE",
    )
    slm_attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = Field(
        default="eager",
        alias="SLM_ATTN_IMPLEMENTATION",
    )
    llama_n_ctx: int = Field(default=2048, alias="LLAMA_N_CTX")
    llama_n_threads: int | None = Field(default=None, alias="LLAMA_N_THREADS")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=3, alias="TOP_K")

    max_new_tokens: int = Field(default=50, alias="MAX_NEW_TOKENS") # max tokens in generated answer was 180 reduced to 50 to fit in GPU memory with the Phi-3.5-mini-instruct model
    temperature: float = Field(default=0.2, alias="TEMPERATURE")

    @field_validator("slm_model_path", mode="before")
    @classmethod
    def empty_model_path_to_none(cls, value: object) -> object:
        if value in ("", None):
            return None
        return value

    @field_validator("gguf_model_path", mode="before")
    @classmethod
    def empty_gguf_path_to_none(cls, value: object) -> object:
        if value in ("", None):
            return None
        return value


def get_settings() -> Settings:
    return Settings()
