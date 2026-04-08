# Changelog

All notable changes to Cortex v2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **HTTP MCP transport**: `cortex serve --transport mcp-http` runs the MCP
  server on streamable-http so multiple clients (Claude Code, dashboard,
  CLI, REST API) can share a single long-lived server. Wires up the
  previously-orphaned `run_http()` function.
- **Global `--direct` CLI flag**: bypasses the MCP HTTP routing and opens
  the graph store directly. Use it when the MCP server is unreachable or
  for bulk admin work where you want the CLI to own the lock.
- **Three new MCP admin tools** exposed to localhost-bound HTTP clients:
  - `cortex_query_trail` — recent query log (default limit 50, max 1000)
  - `cortex_graph_data` — Cytoscape-shaped aggregation for dashboards
  - `cortex_list_entities` — entity listing with optional type filter
- **`StoreLockedError`** exception with `holder_pid`, `holder_cmdline`,
  `is_stale`, `is_pid_reuse`, `db_path`, `marker_path`, and a
  copy-pasteable cleanup hint embedded in the error message.
- **PID marker file** at `~/.cortex/graph.db.lock` so concurrent open
  attempts identify the conflicting process by PID and command line,
  with stale-marker and PID-reuse detection.
- **`cortex register --legacy-stdio`**: opt-in to the pre-Phase-2 stdio
  MCP registration. Default is now HTTP transport.
- **`mcp_server_url` config field** (env var `CORTEX_MCP_SERVER_URL`) —
  where the dashboard, CLI, and REST API connect for HTTP MCP.
- **Dashboard `error.html` template** rendered when MCP is unreachable,
  with an actionable hint pointing at `cortex serve --transport mcp-http`.
- **LaunchAgent plist template** (`ai.abbacus.cortex.mcp`) for
  auto-starting the MCP HTTP server on macOS login. See
  `RESUME_INSTRUCTIONS.md` for the XML and `launchctl` commands.
- **`tests/integration/`** suite with real subprocess end-to-end tests
  verifying concurrent dashboard + Claude Code + CLI access against a
  single MCP HTTP server, including `lsof` verification that only the
  MCP server PID holds `graph.db`.
- **`RESUME_INSTRUCTIONS.md`** documenting the production switch
  sequence, rollback path, and troubleshooting for the Claude Code MCP
  session staleness issue.

### Changed

- **Dashboard rewritten** as a thin MCP HTTP client. No more direct
  `Store(config)` / `PipelineOrchestrator` / `RetrievalEngine` /
  `LearningLoop` instantiations. Every read and write flows through the
  MCP HTTP server. Atomicity is preserved because the MCP server still
  owns both `GraphStore` and `ContentStore` in one process.
- **CLI commands route through MCP by default** (12 commands: capture,
  search, read, list, status, context, dossier, graph, synthesize,
  entities, pipeline, reason). Use `cortex --direct <cmd>` to bypass.
- **Bootstrap commands stay direct**: `init`, `setup`, `import-v1`, and
  `import-vault` always open the store directly because they run before
  any MCP server can exist.
- **`run_http()` admin tool gating**: localhost-bound servers
  (127.0.0.1, localhost, ::1) expose admin tools; non-localhost binds
  disable them to preserve the security boundary for untrusted agents.
- **`Store.close()`** now closes the graph store too (was leaking the
  RocksDB lock until Python garbage collection).
- **`GraphStore.close()`** drops the pyoxigraph reference and forces gc
  to release the underlying RocksDB lock immediately.
- **Migrated to canonical `streamable_http_client`** API. The deprecated
  `streamablehttp_client` alias has been removed from the codebase,
  eliminating ~78 `DeprecationWarning` entries per test run. Added a
  thin `_http_client_session` helper that wraps `httpx.AsyncClient`
  lifecycle to preserve the per-call session pattern.
- **`CortexMCPClient._call` / `list_tools`** use a sentinel guard so
  that under rare `anyio`/task-group teardown conditions, control flow
  that would previously reach `return` with `result` unbound now
  surfaces as a clean `MCPConnectionError` instead of crashing with
  `UnboundLocalError`.

### Fixed

- **Silent data-loss bug** (commit `a30db40`): the previous "fix"
  silently fell back to an empty in-memory Oxigraph when the lock was
  contested, silently dropping all writes from the losing process. Now
  raises `StoreLockedError` with clear holder identification.
- **OSError discrimination**: permission errors and other non-lock
  failures during graph DB init were being misreported as lock errors.
  Now substring-matches RocksDB's specific lock-error variants before
  treating an `OSError` as a lock conflict.
- **PID reuse race** (Weak Point #2): when the OS has recycled a
  marker's PID for an unrelated process, the error now flags
  `is_pid_reuse=True` and warns that the actual holder cannot be
  identified. Compares the running PID's cmdline against the marker's
  recorded cmdline before trusting it.
- **Mid-write atomicity** (Weak Point #11): `Store.create()`'s rollback
  path is now explicitly tested — if `ContentStore.insert()` fails
  after `GraphStore.create_object()` succeeds, the graph write is
  rolled back via `delete_object()` so no orphan triples leak.
- **Unreadable marker file** (`chmod 000 graph.db.lock`): now produces
  a specific error message ("marker exists but is unreadable. Check
  its permissions.") instead of the misleading "no marker file found".
- **`cortex serve --transport mcp-http` bind failures**: privileged
  ports (80), address-in-use, and other `OSError` variants now exit
  cleanly with an actionable message instead of a Python traceback.

### Test count progression

| Milestone                                           | Tests |
| --------------------------------------------------- | ----- |
| Pre-A+D                                             |   549 |
| After Phase 1+2 (A+D-http)                          |   673 |
| After Phase 3 (CLI as MCP client)                   |   716 |
| After Bundle 1 (hardening)                          |   728 |
| After Bundle 2 (intended-failure tests)             |   741 |
| After Bundle 3 (unreadable marker UX)               |   742 |
| After Bundle 4 (deprecation fix + small gaps)       |   757 |
| After Bundle 5 (validation closure)                 |   783 |

### Migration guide

If you're upgrading from a pre-2026-04-07 install:

1. Stop any running `cortex serve --transport stdio` processes.
2. Edit `~/.claude.json` to use HTTP transport:
   ```json
   "cortex": { "type": "http", "url": "http://127.0.0.1:1314/mcp" }
   ```
3. Optionally install the LaunchAgent (see `RESUME_INSTRUCTIONS.md`):
   ```bash
   launchctl load ~/Library/LaunchAgents/ai.abbacus.cortex.mcp.plist
   ```
4. Or run the MCP HTTP server manually whenever you need it:
   ```bash
   cortex serve --transport mcp-http --host 127.0.0.1 --port 1314
   ```
5. Restart Claude Code.

### Known limitations

- **Claude Code MCP session staleness**: when the MCP HTTP server
  restarts, Claude Code's MCP client holds a stale session ID until you
  restart Claude Code itself. This is upstream Claude Code behavior,
  not a Cortex bug. Workaround: restart Claude Code after intentional
  MCP server restarts. The dashboard and CLI do *not* have this issue
  because they use per-call MCP sessions.
- **`cortex pipeline --batch`** requires raw SQL access and therefore
  bypasses MCP routing. Run it with `cortex --direct pipeline --batch`,
  or temporarily stop the MCP server.
- **REST API still opens `graph.db` directly**: `cortex serve
  --transport http` (the REST API on `transport/api/server.py`) is the
  third entry point and has not yet been converged to the MCP-routed
  pattern. Running it simultaneously with the LaunchAgent MCP server
  will fail with the lock error. Will be addressed in a future Phase 4.
