FROM python:3.12-slim AS base

WORKDIR /app

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY ontology/ ontology/
COPY llms.txt ./

# Install dependencies
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Create data directory
RUN mkdir -p /data

ENV CORTEX_DATA_DIR=/data
ENV CORTEX_HOST=0.0.0.0
ENV CORTEX_PORT=1314

EXPOSE 1314

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:1314/health')" || exit 1

# Default: run HTTP API server
CMD ["uv", "run", "cortex", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "1314"]
