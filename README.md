# Cortex

Cognitive knowledge system with formal ontology, reasoning, and intelligence serving.

Cortex captures knowledge objects (decisions, lessons, fixes, sessions, research, ideas), classifies them with an OWL-RL ontology, discovers relationships, reasons over the graph, and serves intelligence through hybrid retrieval.

## Quick Start

```bash
# Install
pip install .
# or
uv sync

# Initialize
cortex init

# Setup (interactive)
cortex setup

# Capture knowledge
cortex capture "Fix: Neo4j pool exhaustion" --type fix --content "Root cause was..."

# Search
cortex search "connection pool"

# Get context (token-efficient briefing)
cortex context "Neo4j"

# Build a dossier
cortex dossier "Neo4j"

# Start the MCP server (for Claude Code / Cursor)
cortex serve

# Start the web dashboard
cortex dashboard

# Start the REST API
cortex serve --transport http
```

## Docker

```bash
docker compose up -d
# Dashboard at http://localhost:1314
```

## Architecture

- **Ontology**: OWL-RL formal ontology with 8 knowledge types and 8 relationship types
- **Storage**: Oxigraph (RDF/SPARQL) + SQLite (FTS5/BM25) dual-write
- **Pipeline**: Classify → Extract entities → Link → Enrich → Reason
- **Retrieval**: Hybrid keyword + semantic + graph-boosted ranking
- **Serving**: 5 presentation modes (briefing, dossier, document, synthesis, alert)
- **Transports**: MCP (stdio + HTTP), REST API, Web Dashboard

## CLI Commands

| Command | Description |
|---------|-------------|
| `cortex init` | Initialize data directory and stores |
| `cortex setup` | Interactive setup wizard |
| `cortex capture` | Capture a knowledge object |
| `cortex search` | Full-text search |
| `cortex read` | Read object in full |
| `cortex list` | List objects with filters |
| `cortex status` | Health and counts |
| `cortex context` | Briefing mode (summaries) |
| `cortex dossier` | Entity-centric intelligence brief |
| `cortex graph` | Show object relationships |
| `cortex synthesize` | Cross-document synthesis |
| `cortex entities` | List resolved entities |
| `cortex register` | Register MCP with Claude Code |
| `cortex serve` | Start MCP or HTTP server |
| `cortex dashboard` | Start web dashboard |
| `cortex import-v1` | Import from Cortex v1 database |
| `cortex import-vault` | Import from Obsidian vault |

## MCP Tools

14 tools for AI agent integration:

**Public**: search, context, dossier, read, capture, link, feedback, graph, list
**Admin**: status, synthesize, delete, export, safety_check

## Knowledge Types

decision, lesson, fix, session, research, source, synthesis, idea

## Relationship Types

causedBy, contradicts (symmetric), supports, supersedes (transitive), dependsOn, ledTo (inverse of causedBy), implements, mentions

## Configuration

Set via environment variables (prefix `CORTEX_`) or `.env` file:

```env
CORTEX_DATA_DIR=~/.cortex
CORTEX_LLM_MODEL=claude-sonnet-4-20250514
CORTEX_LLM_API_KEY=sk-...
CORTEX_PORT=1314
```

See `.env.example` for all options.

## License

Copyright Abbacus Group.
