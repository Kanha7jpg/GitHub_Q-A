from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from app.config import Settings


def _is_valid_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as model_file:
            return model_file.read(4) == b"GGUF"
    except OSError:
        return False


def ensure_quantized_model(settings: Settings, emit: callable) -> None:
    """Ensure GGUF exists for llama.cpp-capable startup, quantizing on demand."""
    if settings.slm_backend not in {"auto", "llama_cpp"}:
        return

    if settings.gguf_model_path is None:
        if settings.slm_backend == "llama_cpp":
            raise RuntimeError("SLM_BACKEND=llama_cpp requires GGUF_MODEL_PATH")
        return

    gguf_path = settings.gguf_model_path.expanduser().resolve()
    if gguf_path.exists() and _is_valid_gguf(gguf_path):
        emit(f"startup: quantized model found at {gguf_path}")
        return

    emit("startup: quantized model missing or invalid; starting auto-quantization")

    project_root = Path(__file__).resolve().parent.parent
    quantize_script = project_root / "quantize.py"
    if not quantize_script.exists():
        raise RuntimeError(f"quantize.py not found at {quantize_script}")

    full_name = f"{gguf_path.stem}-f16-auto.gguf"
    cmd = [
        sys.executable,
        str(quantize_script),
        "--output-dir",
        str(gguf_path.parent),
        "--quantized-name",
        gguf_path.name,
        "--full-name",
        full_name,
        "--backend",
        settings.slm_backend,
    ]

    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        raise RuntimeError(f"auto-quantization failed with exit code {result.returncode}")

    if not gguf_path.exists() or not _is_valid_gguf(gguf_path):
        raise RuntimeError(
            f"auto-quantization finished but GGUF is missing/invalid at {gguf_path}"
        )

    emit(f"startup: auto-quantization complete at {gguf_path}")
