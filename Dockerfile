FROM python:3.12-slim AS base

WORKDIR /app

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY src/ src/
COPY ontology/ ontology/
COPY llms.txt ./

# Install dependencies
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Create data directory and initialize
RUN mkdir -p /data

ENV CORTEX_DATA_DIR=/data
ENV CORTEX_HOST=0.0.0.0
ENV CORTEX_PORT=1314

EXPOSE 1314

# Health check — JSON-RPC ping to MCP endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request,json; r=urllib.request.Request('http://localhost:1314/mcp',data=json.dumps({'jsonrpc':'2.0','id':1,'method':'ping'}).encode(),headers={'Content-Type':'application/json','Accept':'application/json, text/event-stream'},method='POST'); urllib.request.urlopen(r)" || exit 1

# Init data dir if empty (handles mounted volumes), then start server
CMD ["sh", "-c", "uv run cortex init 2>/dev/null; uv run cortex serve --transport mcp-http --host 0.0.0.0 --port 1314"]
