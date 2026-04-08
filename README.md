# Cortex

Cognitive knowledge system with formal ontology, reasoning, and intelligence serving.

Cortex captures knowledge objects (decisions, lessons, fixes, sessions, research, ideas), classifies them with an OWL-RL ontology, discovers relationships, reasons over the graph, and serves intelligence through hybrid retrieval.

## Architecture

Post-Phase-3 Cortex runs as a single **MCP HTTP server** that owns the graph store. Claude Code, the dashboard, the CLI, and the REST API are all HTTP clients of that one server. This lets them run simultaneously without lock fights.

```
┌───────────────┐    ┌────────────┐    ┌─────────────┐
│ Claude Code   │    │  Dashboard │    │     CLI     │
│ (MCP client)  │    │ (browser)  │    │  (terminal) │
└───────┬───────┘    └─────┬──────┘    └──────┬──────┘
        │                  │                  │
        │ HTTP JSON-RPC    │ HTTP MCP         │ HTTP MCP (default)
        │                  │                  │ direct (--direct)
        ▼                  ▼                  ▼
        ┌──────────────────────────────────────┐
        │   cortex serve --transport mcp-http  │
        │   (canonical MCP HTTP server)        │
        │   PID-locked owner of graph.db       │
        └──────────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │  ~/.cortex/                 │
            │    graph.db   (Oxigraph)    │
            │    cortex.db  (SQLite WAL)  │
            └─────────────────────────────┘
```

- **Ontology**: OWL-RL formal ontology with 8 knowledge types and 8 relationship types
- **Storage**: Oxigraph (RDF/SPARQL) + SQLite (FTS5/BM25) dual-write
- **Pipeline**: Classify → Extract entities → Link → Enrich → Reason
- **Retrieval**: Hybrid keyword + semantic + graph-boosted ranking
- **Serving**: 5 presentation modes (briefing, dossier, document, synthesis, alert)
- **Transports**: MCP (stdio + HTTP), REST API, Web Dashboard

## Quick Start

```bash
# Install
pip install .
# or
uv sync

# Bootstrap — runs before any MCP server can exist
cortex init
cortex setup
```

### Three-terminal flow

```bash
# Terminal 1 — the canonical MCP HTTP server (or use the LaunchAgent below)
cortex serve --transport mcp-http --host 127.0.0.1 --port 1314

# Terminal 2 — the web dashboard (a thin MCP HTTP client)
cortex dashboard --port 1315
# → http://localhost:1315

# Terminal 3 (or Claude Code) — CLI commands route through MCP by default
cortex list
cortex search "sqlite"
cortex capture "Fix: Neo4j pool exhaustion" --type fix --content "Root cause was..."
cortex context "Neo4j"
cortex dossier "Neo4j"
```

### `--direct` escape hatch

By default, CLI commands forward to the running MCP server so they coexist with
the dashboard and Claude Code. If the MCP server is unreachable, or you want
the CLI to own the lock directly (for example, to run `cortex pipeline
--batch`), prefix commands with `--direct`:

```bash
# Works even when the MCP server is down
cortex --direct list

# Required for bulk SQL-backed operations
cortex --direct pipeline --batch
```

Bootstrap commands (`init`, `setup`, `import-v1`, `import-vault`) always run
directly — they execute before any MCP server can exist.

### LaunchAgent (macOS)

Cortex ships two LaunchAgent templates under `deploy/` so the MCP HTTP
server and the dashboard can both auto-start on login.

**Install both agents** (substitute your username via `sed`):

```bash
sed 's|YOURUSER|'"$USER"'|g' deploy/ai.abbacus.cortex.mcp.plist \
  > ~/Library/LaunchAgents/ai.abbacus.cortex.mcp.plist
sed 's|YOURUSER|'"$USER"'|g' deploy/ai.abbacus.cortex.dashboard.plist \
  > ~/Library/LaunchAgents/ai.abbacus.cortex.dashboard.plist

launchctl load ~/Library/LaunchAgents/ai.abbacus.cortex.mcp.plist
launchctl load ~/Library/LaunchAgents/ai.abbacus.cortex.dashboard.plist
```

After login, the MCP server runs on `http://127.0.0.1:1314/mcp` and the
dashboard on `http://127.0.0.1:1315/`. Both use `KeepAlive=true`, so
they auto-restart on crash.

**Startup race note**: launchd does not guarantee load order. At cold
boot, the dashboard may briefly probe an MCP server that isn't up yet,
exit with "Cannot reach MCP server", and get restarted by launchd.
Expect one or two such entries in `~/.cortex/dashboard.err` at login —
they're harmless.

**Uninstall**:

```bash
launchctl unload ~/Library/LaunchAgents/ai.abbacus.cortex.dashboard.plist
launchctl unload ~/Library/LaunchAgents/ai.abbacus.cortex.mcp.plist
rm ~/Library/LaunchAgents/ai.abbacus.cortex.{mcp,dashboard}.plist
```

**Recovery from a crashed MCP server**: if the MCP server process is
`kill -9`'d or the machine is hard-rebooted, a stale `graph.db.lock`
marker plus a RocksDB `LOCK` file may remain. The next `GraphStore`
open auto-detects and cleans them up when the marker's PID is verified
dead. For stubborn cases, run `cortex doctor unlock` — see the
[Known limitations](#known-limitations) section.

### Claude Code integration

Edit `~/.claude.json` so the `cortex` MCP entry uses HTTP transport:

```json
"cortex": {
  "type": "http",
  "url": "http://127.0.0.1:1314/mcp"
}
```

Restart Claude Code. If you intentionally restart the MCP server later, you
also have to restart Claude Code to clear its stale MCP session ID — see
**Known limitations** below.

## Docker

```bash
docker compose up -d
# Dashboard at http://localhost:1314
```

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

17 tools for AI agent integration (localhost-bound HTTP exposes all; non-localhost binds expose only the public set):

**Public**: `cortex_search`, `cortex_context`, `cortex_dossier`, `cortex_read`, `cortex_capture`, `cortex_link`, `cortex_feedback`, `cortex_graph`, `cortex_list`, `cortex_classify`, `cortex_pipeline`
**Admin**: `cortex_status`, `cortex_synthesize`, `cortex_delete`, `cortex_reason`, `cortex_query_trail`, `cortex_graph_data`, `cortex_list_entities`

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
CORTEX_MCP_SERVER_URL=http://127.0.0.1:1314/mcp
```

See `.env.example` for all options.

## Known limitations

- **Claude Code MCP session staleness**: when the MCP HTTP server
  restarts (LaunchAgent reload, manual `launchctl unload && load`,
  server crash + KeepAlive restart), Claude Code's MCP client keeps its
  old session ID and subsequent tool calls fail with `Session not
  found`. This is upstream Claude Code behavior, not a Cortex bug.
  Workaround: restart Claude Code (`Cmd+Q` and re-open) after any
  intentional MCP server restart. `claude --resume` restores your
  conversation. The dashboard and CLI do **not** have this issue
  because they use per-call MCP sessions.
- **`cortex pipeline --batch`** requires raw SQL access and therefore
  bypasses MCP routing. Run it with `cortex --direct pipeline --batch`,
  or temporarily stop the MCP server.

## Recovering from a crashed MCP server

If the MCP server process is killed hard (`kill -9`, power loss, kernel
panic), it can leave behind both `~/.cortex/graph.db.lock` (Cortex's PID
marker) and `~/.cortex/graph.db/LOCK` (RocksDB's internal lock file).

Most of the time you don't need to do anything — the next `GraphStore`
open detects the stale marker, verifies the recorded PID is really dead,
removes both files, and retries automatically. A single `INFO` line
("Auto-recovered stale lock") appears in the log.

For cases where auto-recovery can't confirm safety (e.g. `PermissionError`
from a cross-user process), use the doctor command:

```bash
cortex doctor unlock              # normal cleanup (refuses to unlock a live holder)
cortex doctor unlock --dry-run    # report what would be removed
cortex doctor unlock --force      # bypass the live-holder check (advanced)
```

`cortex dashboard --spawn-mcp` also auto-launches the MCP server as a
subprocess if none is running, so a crashed or unstarted MCP server
doesn't block dashboard use.

See `CHANGELOG.md` for the full release history.

## License

Copyright Abbacus Group.
