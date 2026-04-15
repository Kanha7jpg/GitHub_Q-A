# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS py-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"
WORKDIR /app

FROM py-base AS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

FROM deps AS model-builder
ARG MODEL_NAME=microsoft/Phi-3.5-mini-instruct
ARG QUANTIZED_DIR=/opt/models/phi3_quantized
ARG QUANTIZED_NAME=phi-3.5-q4_k_m-local.gguf
COPY . .
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_NAME}')" && \
    python quantize.py \
      --output-dir "${QUANTIZED_DIR}" \
      --quantized-name "${QUANTIZED_NAME}" \
      --full-name "phi-3.5-f16-build.gguf" \
      --backend llama_cpp \
      --skip-env-update && \
    test -f "${QUANTIZED_DIR}/${QUANTIZED_NAME}"

FROM deps AS quantizer
COPY . .
# Build this target for use as a Kubernetes initContainer image.
CMD ["sh", "-lc", "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/Phi-3.5-mini-instruct')\"; python quantize.py --output-dir /models/phi3_quantized --quantized-name phi-3.5-q4_k_m-local.gguf --full-name phi-3.5-f16-init.gguf --backend llama_cpp --skip-env-update"]

FROM py-base AS runtime-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deps /opt/venv /opt/venv
COPY . .
EXPOSE 8000

FROM runtime-base AS runtime-preloaded
COPY --from=model-builder /opt/models /opt/models
ENV SLM_MODEL_NAME=microsoft/Phi-3.5-mini-instruct \
    SLM_LOCAL_FILES_ONLY=true \
    SLM_BACKEND=llama_cpp \
    GGUF_MODEL_PATH=/opt/models/phi3_quantized/phi-3.5-q4_k_m-local.gguf \
    LLAMA_N_CTX=2048
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime-base AS runtime
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
