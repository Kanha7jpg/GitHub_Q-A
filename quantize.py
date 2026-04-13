"""
quantize.py  —  One-shot Phi-3.5-mini-instruct → GGUF Q4_K_M
---------------------------------------------------------------
Place this file in your project root (next to requirements.txt).
Run with:
        python quantize.py

What it does:
    1. Reads .env to resolve SLM_MODEL_PATH (or discovers snapshot from HF cache)
    2. Clones llama.cpp into tools/llama.cpp
    3. Converts your local HF model to a full-precision GGUF
    4. Quantizes that GGUF to Q4_K_M
    5. Deletes the intermediate full-precision GGUF
    6. Updates .env with GGUF_MODEL_PATH for plug-and-play app startup
"""

import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SLM_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"

# Output directory for the final GGUF (inside your project by default)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"

# Final quantized model filename
DEFAULT_QUANTIZED_NAME = "phi-3.5-q4_k_m.gguf"

# Intermediate full-precision GGUF (deleted after quantization)
DEFAULT_FULL_GGUF_NAME = "phi-3.5-f16.gguf"

# llama.cpp will be cloned here (inside the project, gitignored)
LLAMA_CPP_DIR = PROJECT_ROOT / "tools" / "llama.cpp"

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command, streaming output live."""
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n✗ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def check_cmake() -> bool:
    return shutil.which("cmake") is not None


def get_build_toolchain() -> tuple[str | None, str | None]:
    """Return (generator, reason) for building llama.cpp on the current host."""
    if os.name != "nt":
        return None, None

    if shutil.which("ninja"):
        return "Ninja", None

    if shutil.which("cl") and shutil.which("nmake"):
        return "NMake Makefiles", None

    if shutil.which("gcc") and (shutil.which("mingw32-make") or shutil.which("make")):
        return "MinGW Makefiles", None

    # Fallback to Visual Studio generator when Build Tools are installed,
    # even if cl.exe is not on PATH in this shell.
    return "Visual Studio 17 2022", None


def find_quantize_binary(llama_dir: Path) -> Path | None:
    """Find llama-quantize binary after build."""
    candidates = [
        llama_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_dir / "build" / "bin" / "llama-quantize.exe",
        llama_dir / "build" / "Release" / "llama-quantize.exe",
        llama_dir / "llama-quantize.exe",
        llama_dir / "llama-quantize",                  # Linux/Mac
        llama_dir / "build" / "bin" / "Release" / "llama-quantize",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def is_valid_gguf(path: Path) -> bool:
    """Check GGUF magic bytes."""
    try:
        with path.open("rb") as f:
            magic = f.read(4)
        return magic == b"GGUF"
    except Exception:
        return False


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_snapshot_from_env() -> Path | None:
    env_values = parse_env_file(ENV_PATH)
    slm_path = env_values.get("SLM_MODEL_PATH")
    if not slm_path:
        return None

    candidate = Path(slm_path).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()
    return None


def resolve_snapshot_from_cache() -> Path | None:
    env_values = parse_env_file(ENV_PATH)
    model_name = env_values.get("SLM_MODEL_NAME", DEFAULT_SLM_MODEL_NAME)
    model_slug = model_name.replace("/", "--")
    snapshots_dir = DEFAULT_HF_CACHE_ROOT / f"models--{model_slug}" / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        [path for path in snapshots_dir.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if list(candidate.glob("*.safetensors")):
            return candidate.resolve()
    return None


def resolve_hf_snapshot() -> Path | None:
    from_env = resolve_snapshot_from_env()
    if from_env is not None:
        return from_env
    return resolve_snapshot_from_cache()


def upsert_env_value(path: Path, key: str, value: str) -> None:
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    prefix = f"{key}="
    for idx, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[idx] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize local Phi-3.5-mini-instruct snapshot to GGUF Q4_K_M"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for GGUF files (absolute or project-relative path).",
    )
    parser.add_argument(
        "--quantized-name",
        default=DEFAULT_QUANTIZED_NAME,
        help="Final quantized GGUF filename.",
    )
    parser.add_argument(
        "--full-name",
        default=DEFAULT_FULL_GGUF_NAME,
        help="Intermediate F16 GGUF filename.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "llama_cpp", "transformers"],
        default="auto",
        help="Value to write into SLM_BACKEND when updating .env.",
    )
    parser.add_argument(
        "--skip-env-update",
        action="store_true",
        help="Do not update GGUF_MODEL_PATH/SLM_BACKEND in .env.",
    )
    return parser.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    quantized_name = args.quantized_name
    full_gguf_name = args.full_name

    print("=" * 60)
    print("  Phi-3.5-mini-instruct  →  GGUF Q4_K_M quantizer")
    print("=" * 60)

    hf_snapshot = resolve_hf_snapshot()

    # ── Validate snapshot path ────────────────────────────────────────────────
    if hf_snapshot is None or not hf_snapshot.exists():
        print("\n✗ Could not resolve a local Phi snapshot.")
        print("  Set SLM_MODEL_PATH in .env to your local snapshot directory.")
        sys.exit(1)

    safetensors = list(hf_snapshot.glob("*.safetensors"))
    if not safetensors:
        print(f"\n✗ No .safetensors files found in snapshot directory.")
        sys.exit(1)

    print(f"\n✓ Using HF snapshot:\n  {hf_snapshot}")
    print(f"\n✓ Snapshot found  ({len(safetensors)} safetensors file(s))")
    output_dir.mkdir(parents=True, exist_ok=True)

    full_gguf_path = output_dir / full_gguf_name
    quantized_path = output_dir / quantized_name

    # ── Skip if already quantized ─────────────────────────────────────────────
    if quantized_path.exists() and is_valid_gguf(quantized_path):
        size_gb = quantized_path.stat().st_size / 1e9
        print(f"\n✓ Quantized model already exists ({size_gb:.1f} GB):")
        print(f"  {quantized_path.resolve()}")
        if not args.skip_env_update:
            upsert_env_value(ENV_PATH, "GGUF_MODEL_PATH", str(quantized_path.resolve()))
            upsert_env_value(ENV_PATH, "SLM_BACKEND", args.backend)
            print(f"\n  Updated .env: GGUF_MODEL_PATH={quantized_path.resolve()}")
            print(f"  Updated .env: SLM_BACKEND={args.backend}")
        print("\n  Nothing to do. See instructions at the bottom.")
        print_next_steps(quantized_path)
        return

    # ── Step 1: Clone llama.cpp ───────────────────────────────────────────────
    print("\n── Step 1 of 4: Setting up llama.cpp ──")
    LLAMA_CPP_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not LLAMA_CPP_DIR.exists():
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp",
            str(LLAMA_CPP_DIR),
        ])
    else:
        print(f"  llama.cpp already cloned at {LLAMA_CPP_DIR}")

    # ── Step 2: Install Python deps ───────────────────────────────────────────
    print("\n── Step 2 of 4: Installing llama.cpp Python requirements ──")
    requirements_file = LLAMA_CPP_DIR / "requirements.txt"
    if requirements_file.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"])
    else:
        # Newer llama.cpp uses gguf package directly
        run([sys.executable, "-m", "pip", "install", "gguf", "numpy", "-q"])

    # ── Step 3: Convert HF → full-precision GGUF ─────────────────────────────
    print("\n── Step 3 of 4: Converting HF model to GGUF (F16) ──")
    print("  This reads ~7 GB from disk — expect 5–15 minutes...\n")

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Older llama.cpp used a different name
        convert_script = LLAMA_CPP_DIR / "convert.py"

    if not convert_script.exists():
        print("✗ Could not find convert_hf_to_gguf.py or convert.py in llama.cpp")
        sys.exit(1)

    run([
        sys.executable, str(convert_script),
        str(hf_snapshot),
        "--outfile", str(full_gguf_path),
        "--outtype", "f16",
    ])

    if not full_gguf_path.exists() or not is_valid_gguf(full_gguf_path):
        print("\n✗ Conversion produced no valid GGUF file.")
        sys.exit(1)

    size_gb = full_gguf_path.stat().st_size / 1e9
    print(f"\n✓ Full-precision GGUF written ({size_gb:.1f} GB)")

    # ── Step 4: Build llama-quantize and quantize ─────────────────────────────
    print("\n── Step 4 of 4: Quantizing to Q4_K_M ──")

    quantize_bin = find_quantize_binary(LLAMA_CPP_DIR)

    if quantize_bin is None:
        if not check_cmake():
            print("\n✗ cmake not found. Cannot build llama-quantize.")
            print("  Install CMake from https://cmake.org/download/")
            print("  or via:  winget install Kitware.CMake")
            sys.exit(1)

        generator, reason = get_build_toolchain()
        if reason is not None:
            print(f"\n✗ {reason}")
            if os.name == "nt":
                print(
                    "  Recommended install command:\n"
                    "  winget install --id Microsoft.VisualStudio.2022.BuildTools "
                    "--override \"--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\""
                )
            sys.exit(1)

        print("  Building llama.cpp (this takes a few minutes)...")
        build_dir = LLAMA_CPP_DIR / "build"
        configure_cmd = ["cmake", "-B", str(build_dir), str(LLAMA_CPP_DIR)]
        if generator:
            configure_cmd = ["cmake", "-G", generator]
            if generator.startswith("Visual Studio"):
                configure_cmd.extend(["-A", "x64"])
            configure_cmd.extend(["-B", str(build_dir), str(LLAMA_CPP_DIR)])
        run(configure_cmd)
        run(["cmake", "--build", str(build_dir), "--config", "Release",
             "--target", "llama-quantize"])

        quantize_bin = find_quantize_binary(LLAMA_CPP_DIR)

    if quantize_bin is None:
        print("\n✗ Could not locate llama-quantize binary after build.")
        sys.exit(1)

    print(f"  Using quantizer: {quantize_bin}")
    print("  Quantizing to Q4_K_M — expect 5–10 minutes...\n")

    run([
        str(quantize_bin),
        str(full_gguf_path),
        str(quantized_path),
        "Q4_K_M",
    ])

    # ── Cleanup intermediate GGUF ─────────────────────────────────────────────
    if full_gguf_path.exists():
        print(f"\n  Deleting intermediate F16 GGUF ({size_gb:.1f} GB)...")
        full_gguf_path.unlink()
        print("  ✓ Deleted")

    # ── Done ──────────────────────────────────────────────────────────────────
    if quantized_path.exists() and is_valid_gguf(quantized_path):
        final_size = quantized_path.stat().st_size / 1e9
        if not args.skip_env_update:
            upsert_env_value(ENV_PATH, "GGUF_MODEL_PATH", str(quantized_path.resolve()))
            upsert_env_value(ENV_PATH, "SLM_BACKEND", args.backend)
        print(f"\n{'=' * 60}")
        print(f"  ✓ Quantization complete!  ({final_size:.1f} GB)")
        print(f"{'=' * 60}")
        if not args.skip_env_update:
            print(f"  Updated .env: GGUF_MODEL_PATH={quantized_path.resolve()}")
            print(f"  Updated .env: SLM_BACKEND={args.backend}")
        else:
            print("  Skipped .env update (--skip-env-update)")
        print_next_steps(quantized_path)
    else:
        print("\n✗ Quantized file not found or invalid.")
        sys.exit(1)


def print_next_steps(quantized_path: Path) -> None:
    abs_path = quantized_path.resolve()
    print(f"""
Next steps
──────────
1. Install llama-cpp-python (CPU build):

     pip install llama-cpp-python

2. Add to your .env:

     GGUF_MODEL_PATH={abs_path}

3. Restart your FastAPI server. With SLM_BACKEND=auto,
     the app will use GGUF_MODEL_PATH automatically when present.

4. Load time should drop significantly,
   and each /ask query should respond in 2–5 minutes on CPU.
""")


if __name__ == "__main__":
    main()
