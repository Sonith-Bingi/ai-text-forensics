FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install --no-cache-dir .


FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app
COPY src ./src
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e .

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "forensics.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
