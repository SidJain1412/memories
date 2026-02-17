# ---- Build stage: core deps + model cache ----
FROM python:3.11-slim AS builder-core

ARG ENABLE_CLOUD_SYNC=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optionally install cloud sync dependencies (adds ~80MB)
COPY requirements-cloud.txt .
RUN if [ "$ENABLE_CLOUD_SYNC" = "true" ]; then \
        pip install --no-cache-dir -r requirements-cloud.txt; \
    fi

# Copy embedder so we can pre-download the ONNX model
COPY onnx_embedder.py .

# Pre-download the ONNX model + tokenizer so first startup is fast
RUN python -c "from onnx_embedder import OnnxEmbedder; OnnxEmbedder('all-MiniLM-L6-v2')"

# Strip test/doc bloat from installed packages
RUN find /usr/local/lib/python3.11/site-packages \
    \( -type d -name "tests" -o -name "test" -o -name "__pycache__" \) \
    -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete 2>/dev/null; \
    true

# ---- Build stage: extraction deps ----
FROM builder-core AS builder-extract

COPY requirements-extract.txt .
RUN pip install --no-cache-dir -r requirements-extract.txt

# Strip test/doc bloat from extraction SDK installs
RUN find /usr/local/lib/python3.11/site-packages \
    \( -type d -name "tests" -o -name "test" -o -name "__pycache__" \) \
    -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete 2>/dev/null; \
    true

# ---- Runtime stage: shared base ----
FROM python:3.11-slim AS runtime-base

WORKDIR /app

# Copy application code
COPY onnx_embedder.py .
COPY memory_engine.py .
COPY cloud_sync.py .
COPY runtime_memory.py .
COPY llm_provider.py .
COPY llm_extract.py .
COPY app.py .

RUN mkdir -p /data/backups

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import sys,urllib.request; sys.exit(0 if 200 <= urllib.request.urlopen('http://localhost:8000/health', timeout=3).getcode() < 400 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Runtime target: extract (includes Anthropic/OpenAI SDKs) ----
FROM runtime-base AS extract

COPY --from=builder-extract /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder-extract /usr/local/bin /usr/local/bin
COPY --from=builder-extract /root/.cache/huggingface /root/.cache/huggingface

# ---- Runtime target: core (default, no extraction SDKs) ----
FROM runtime-base AS core

COPY --from=builder-core /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder-core /usr/local/bin /usr/local/bin
COPY --from=builder-core /root/.cache/huggingface /root/.cache/huggingface
