from pathlib import Path
from typing import Literal

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
    slm_attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = Field(
        default="eager",
        alias="SLM_ATTN_IMPLEMENTATION",
    )

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=3, alias="TOP_K")

    max_new_tokens: int = Field(default=50, alias="MAX_NEW_TOKENS") # max tokens in generated answer was 180 reduced to 50 to fit in GPU memory with the Phi-3.5-mini-instruct model
    temperature: float = Field(default=0.2, alias="TEMPERATURE")


def get_settings() -> Settings:
    return Settings()
