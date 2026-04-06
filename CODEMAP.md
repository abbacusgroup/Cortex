# Cortex Code Map

Quick orientation for developers.

## Project Structure

```
cortex/
├── src/cortex/
│   ├── __init__.py              # Version
│   ├── core/
│   │   ├── config.py            # Config loader (env + .env + defaults)
│   │   ├── constants.py         # Global constants, type definitions
│   │   ├── errors.py            # Error hierarchy (14 types)
│   │   └── logging.py           # JSON structured logging
│   ├── ontology/
│   │   └── namespaces.py        # RDF namespace helpers, IRI builders
│   ├── db/
│   │   ├── graph_store.py       # Oxigraph RDF store (CRUD, SPARQL)
│   │   ├── content_store.py     # SQLite store (FTS5, embeddings, config, logs)
│   │   └── store.py             # Unified dual-write sync layer
│   ├── pipeline/
│   │   ├── normalize.py         # LLM classification + embedding generation
│   │   ├── link.py              # Entity resolution + relationship discovery
│   │   ├── enrich.py            # Tier assignment + staleness scoring
│   │   ├── reason.py            # OWL-RL inference (SPARQL CONSTRUCT)
│   │   ├── advanced_reason.py   # Contradictions, patterns, gaps, staleness
│   │   ├── temporal.py          # Version history + state-at-time queries
│   │   ├── templates.py         # Capture templates (session, fix, decision)
│   │   ├── orchestrator.py      # Pipeline coordinator
│   │   └── importer.py          # V1 SQLite + Obsidian vault importers
│   ├── retrieval/
│   │   ├── engine.py            # Hybrid search (keyword + semantic + graph)
│   │   ├── presenters.py        # 5 modes: briefing, dossier, document, synthesis, alert
│   │   ├── graph.py             # Causal chain, contradiction map, neighborhood
│   │   └── learner.py           # Access tracking, tier promotion, weight tuning
│   ├── services/
│   │   └── llm.py               # litellm wrapper, classification, relationship prompts
│   ├── transport/
│   │   ├── mcp/server.py        # MCP server (14 tools, stdio + HTTP)
│   │   └── api/server.py        # REST API (FastAPI, auth, rate limiting)
│   ├── cli/
│   │   └── main.py              # Typer CLI (18 commands)
│   └── dashboard/
│       ├── server.py            # Dashboard server (FastAPI + Jinja2)
│       ├── templates/           # 9 HTML templates
│       └── static/              # CSS + JS (Cytoscape.js)
├── ontology/
│   └── cortex.ttl               # OWL ontology (304 triples)
├── tests/                       # ~490 tests mirroring src/ structure
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── llms.txt                     # Agent onboarding doc
```

## Data Flow

```
Capture → Store (Oxigraph + SQLite)
       → Normalize (LLM classify + embed)
       → Link (entity resolution + relationships)
       → Enrich (tier assignment)
       → Reason (OWL-RL inference)
       → Serve (search, briefing, dossier, synthesis)
```

## Key Design Decisions

1. **Dual store**: Oxigraph for graph/SPARQL, SQLite for content/FTS5. Sync layer keeps both consistent.
2. **OWL-RL**: Formal ontology enables real inference (transitive, symmetric, inverse properties).
3. **Pipeline resilience**: Each stage can fail independently. Partial completion is OK.
4. **No LLM required**: System works without LLM (fallback classification). LLM enhances but isn't required.
5. **Memory tiers**: archive → recall → reflex. Learning loop promotes/demotes based on access.
