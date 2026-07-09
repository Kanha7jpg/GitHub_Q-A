from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import subprocess

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source: str
    text: str


class RepositoryIngestor:
    """Clones a repository and converts supported files into chunks."""

    IGNORED_DIRS = {
        ".git", "__pycache__", ".venv", "venv", 
        "node_modules", "dist", "build",
    }
    
    SUPPORTED_EXTENSIONS = {".py", ".md", ".js", ".jsx", ".ts", ".tsx"}
    
    LANGUAGE_MAP = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.splitters: dict[Language, RecursiveCharacterTextSplitter] = {}
        
        # Initialize language-specific splitters
        for ext, language in self.LANGUAGE_MAP.items():
            if language not in self.splitters:
                self.splitters[language] = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
        
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n", " ", ""],
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def clone_repository(self) -> Path:
        clone_dir = self.settings.clone_dir
        clone_dir.parent.mkdir(parents=True, exist_ok=True)

        source_path = Path(self.settings.repo_url)
        if source_path.exists() and source_path.is_dir():
            logger.info(f"Using local repository at {source_path}")
            return source_path.resolve()

        if (clone_dir / ".git").exists():
            logger.info(f"Updating existing repository at {clone_dir}")
            subprocess.run(
                ["git", "-C", str(clone_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                text=True,
            )
            return clone_dir

        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        logger.info(f"Cloning repository from {self.settings.repo_url}")
        subprocess.run(
            ["git", "clone", "--depth", "1", self.settings.repo_url, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return clone_dir

    def _iter_supported_files(self, repo_path: Path) -> list[Path]:
        files: list[Path] = []

        for root, dirnames, filenames in os.walk(repo_path):
            dirnames[:] = [name for name in dirnames if name not in self.IGNORED_DIRS]

            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files.append(file_path)

        logger.info(f"Found {len(files)} supported files to ingest")
        return files

    @staticmethod
    def _read_text(file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ""

    def _split_file(self, file_path: Path, text: str) -> list[str]:
        if not text:
            return []
        
        suffix = file_path.suffix.lower()
        
        if suffix in self.LANGUAGE_MAP:
            language = self.LANGUAGE_MAP[suffix]
            splitter = self.splitters[language]
        else:
            splitter = self.markdown_splitter
        
        chunks = splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def build_chunk_records(self) -> list[ChunkRecord]:
        repo_path = self.clone_repository()
        files = self._iter_supported_files(repo_path)
        records: list[ChunkRecord] = []

        for file_path in files:
            file_text = self._read_text(file_path)
            relative_source = str(file_path.relative_to(repo_path)).replace("\\", "/")
            chunk_texts = self._split_file(file_path, file_text)

            for idx, chunk_text in enumerate(chunk_texts):
                chunk_id = f"{relative_source}::{idx}"
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source=relative_source,
                        text=chunk_text,
                    )
                )
        
        logger.info(f"Generated {len(records)} chunks from {len(files)} files")
        return records
