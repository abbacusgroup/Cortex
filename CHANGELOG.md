# Changelog

All notable changes to Cortex v2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Distribution Bundle

- **Optional `sentence-transformers` dependency**: moved from core
  `dependencies` to `[project.optional-dependencies] embeddings`.
  `pip install abbacus-cortex` drops from ~2.5 GB to ~200 MB.
  `pip install abbacus-cortex[embeddings]` for semantic search.
- **Embedding model warm-up in `cortex init`**: downloads and caches the
  embedding model during initialization instead of surprising users on
  first search. Reports status: loading, ready, not installed, or failed.
- **`RetrievalEngine._get_embedder()` caching**: was creating a new
  `SentenceTransformer` (~400 MB) on every search call. Now cached with
  instance reuse (same pattern as `NormalizeStage._get_embedder()`).
- **`cortex install` / `cortex uninstall`** — platform-aware background
  service setup. macOS: renders LaunchAgent plist, `launchctl load`.
  Linux: renders systemd user unit, `systemctl --user enable --now`.
  Templates embedded in `install.py` (not read from `deploy/`) because
  `deploy/` is outside `src/cortex/` and not in the pip wheel.
- **Linux systemd user unit templates** in `deploy/cortex-mcp.service`
  and `deploy/cortex-dashboard.service` for manual reference.
- **GitHub Release workflow** (`.github/workflows/release.yml`): triggered
  on `v*` tag push. Reuses `test.yml` as a gate, builds wheel + sdist
  with `uv build`, creates GitHub Release with artifacts attached.
- **README rewritten** for end-user install flow: lead with `pip install`,
  3-step quickstart (init → install → register → use), updated MCP tool
  count from 17 to 22, added service management and troubleshooting
  sections.

### Added — Bundle 10.9

- **SIM ruff family enabled** (10th lint family): 12 sites refactored
  from `try/except/pass` to `contextlib.suppress()`. All in non-critical
  paths (shutdown cleanup, best-effort enrichment). Semantically identical.
- **A.2 diagnostic admin tools**: `cortex_debug_sessions` (session table
  snapshot) and `cortex_debug_memory` (tracemalloc top-10 allocations).
  Opt-in, admin-only, localhost-gated.
- **Security probe tests**: E.3 path traversal (10 tests covering
  `../`, URL-encoded, and nested traversal payloads) and E.2 REST API
  auth bypass (10 tests covering missing key, wrong key, empty key,
  whitespace key, timing-safe comparison). All pure test additions —
  no production code changes.

### Fixed — Bundle 10.9

- **A.2 session-table leak**: enabled stateless HTTP mode
  (`stateless_http=True` in FastMCP). Zombie session count dropped from
  44 after 20 calls to 0. Root cause: upstream MCP SDK never evicts
  terminated sessions from `_server_instances` in stateful mode.
- **CI macOS probe timeout flake**: widened the probe timeout band in
  the test fixture to accommodate slow cold starts on GitHub Actions
  macOS runner.

### Fixed — Bundle 10.8 (BaseExceptionGroup unwrapping in CortexMCPClient)

- **"Unhandled errors in a TaskGroup" misclassification** in
  `transport/mcp/client.py`: when anyio's `TaskGroup` wraps a
  transport error (e.g. `httpx.ReadTimeout`) in a
  `BaseExceptionGroup`, the new `except BaseExceptionGroup` handler
  in both `_call` and `list_tools` unwraps the group via
  `_flatten_exception_group`, picks the most significant leaf via
  `_pick_significant_leaf` (priority: timeout > HTTP status >
  connection > first), and classifies it using a shared
  `_classify_transport_exception` helper. Timeouts that previously
  surfaced as `MCPConnectionError("...unhandled errors in a
  TaskGroup (1 sub-exception)")` now produce a clean
  `MCPTimeoutError("MCP server at ... timed out after 30.0s")`.
  `asyncio.CancelledError` inside a group is detected and re-raised
  to preserve clean cancellation semantics. 9 new tests in
  `tests/transport/test_mcp_client.py`.

### Changed — Bundle 10.8

- **SIM117 refactor** in `transport/mcp/client.py`: combined the
  three nested `async with` pairs in `_http_client_session`, `_call`,
  and `list_tools` into single parenthesized `async with` statements.
  Semantically equivalent per PEP 617. These were the three sites
  that blocked the SIM family from enabling in Bundle 10.5c; now
  `ruff --select SIM117` passes clean on `client.py`.
- **Unified exception classification** in `transport/mcp/client.py`:
  the per-exception-type `except` ladders in `_call` and `list_tools`
  (4 clauses each, duplicated) are replaced by a shared helper
  `_classify_transport_exception` that maps any single transport
  exception to the correct typed error class. The `BaseExceptionGroup`
  handler calls the same helper after unwrapping. Reduces duplication
  and ensures bare and grouped exceptions are classified identically.

### Added — Bundle 10 (SPARQL + templates + ruff expansion)

- **SPARQL string-literal escape helper** (`_sparql_escape_string`) in
  `db/graph_store.py`, Bundle 10.1: implements SPARQL 1.1 §5.4 ECHAR
  rules (backslash first, then `"`, `\n`, `\r`, `\t`). Applied at the
  one user-reachable interpolation site — the `project` filter in
  `list_objects`. Private helper (`_` prefix). Covered by 9 new tests
  in `tests/db/test_graph_store.py` (`TestSparqlEscape` unit tests +
  `TestSparqlInjectionListObjects` adversarial tests with quote,
  backslash, newline, and classic UNION-injection payloads). The other
  4 SPARQL interpolation sites in `graph_store.py` are whitelisted
  (`obj_type` via `CLASS_MAP`, `limit`/`offset` typed `int`) and left
  for a future defense-in-depth pass.
- **Direct Jinja2 template render tests** (`tests/dashboard/test_templates_direct.py`,
  19 tests, Bundle 10.2): closes Weak Point #6 from the original A+D
  plan. Each of the 10 data-bearing dashboard templates gets at least
  one direct render test against a hand-built dict fixture that
  mirrors the MCP client return shape. Covers edge cases —
  `created_at=None`, empty `alerts`/`recent`/`logs`, missing
  `content`/`relationships`. `settings.html` deliberately tested
  against a real `CortexConfig` object to match production.
  `TestStaticTemplates` sanity-compiles `create.html` and `graph.html`.
- **`CORTEX_MCP_CLIENT_TIMEOUT_SECONDS` env var** in `cli/main.py`'s
  `_get_mcp_client()` (Bundle 10.6): mirrors the `_get_probe_client`
  pattern from Bundle 9.4. Defaults to 30s for local use; the CI
  workflow now sets it to 60s alongside `CORTEX_PROBE_TIMEOUT_SECONDS`
  so the slow GitHub Actions macOS runner has enough headroom for
  cold-start `cortex capture` calls (which load the sentence-
  transformers embedding model server-side).
- **Test stderr diagnostic helpers** in
  `tests/integration/test_phase3_cli_concurrency.py` (Bundle 10.6):
  `_drain_pipe_nonblocking()` and `_assert_cli_ok()` do a bounded
  non-blocking `os.read` of the MCP server subprocess's stdout+stderr
  via `O_NONBLOCK` on test failure and include the output in
  `pytest.fail()`'s message. Wired into
  `test_capture_then_read_then_list_round_trip`. Next CI failure of
  that shape will surface the actual server-side exception instead of
  the wrapped client view.

### Changed — Bundle 10

- **Ruff lint families expanded** from `["E","F","I","N","W","UP"]` to
  `["B","C4","E","F","I","N","RUF","W","UP"]` — three new families
  enabled across three commits (Bundle 10.5a/b/c):
  - **`B` (flake8-bugbear)**, 45 violations resolved: 41 × B904
    (`raise ... from ...` in except blocks) applied mechanically across
    10 files via a general-purpose agent; `from err` where the except
    had an `as` binding and `from None` otherwise. 2 × B013 (length-one
    tuple literals in `except (X,):`) auto-fixed. 1 × B905
    (`zip(a, b, strict=True)` in cosine similarity). 1 × B007 (unused
    loop variable renamed to `_doc_id`). Zero logic changes, zero
    renamed public symbols.
  - **`C4` (flake8-comprehensions)**, zero violations — config-only
    commit, codebase was already clean.
  - **`RUF`**, 21 violations resolved: 1 × RUF100 (unused
    `noqa: BLE001`), 1 × RUF003 (MULTIPLICATION SIGN `×` → ASCII `x`
    in a comment), 2 × RUF005 (list concatenation → spread unpacking),
    1 × RUF034 (dead `holder_pid=None if False else None` removed),
    16 × RUF059 (unused unpacked variables in test fixtures renamed to
    underscore-prefixed form, targeted not global).
  - **`SIM` attempted and reverted**: 3 × SIM117 sites in
    `transport/mcp/client.py` sit inside the per-call session pattern
    just stabilized in Bundle 9, too risky for a ruff sweep. Deferred
    to a dedicated Bundle 10.7 refactor.
- **`tasks/` added to `.gitignore`** (Bundle 10.4): enforces the
  workspace rule that all session artifacts live in `~/Lab/` rather
  than the cortex repo. Removed the stray
  `~/Lab/cortex/tasks/bug_hunt_2026-04-08_findings.md` that was
  byte-identical to `~/Lab/bug_hunt_2026-04-08.md`.
- **CHANGELOG backfill** (Bundle 10.3 then this entry): Bundle 10.3
  added a Bundle 9 block; this entry backfills the Bundle 10.1–10.6
  changes that were missing at Bundle 10.6 ship time.

### Added — Bundle 10.7 (log growth mitigation, F.4)

- **`cortex doctor logs`** subcommand in `cli/main.py`: inspects and
  rotates the LaunchAgent log files under `~/.cortex/`. Default view
  shows size, line count, last-modified time, and a GREEN / YELLOW /
  RED status badge for each of `mcp-http.log`, `mcp-http.err`,
  `dashboard.log`, `dashboard.err`. `--tail N` shows the last N lines
  of each existing file. `--rotate` copies each non-empty log to
  `<file>.old` and truncates the live file to zero length — safe while
  the LaunchAgent is running because launchd opens stdout/stderr with
  `O_APPEND`, so the server's next write lands at offset 0 without
  needing a restart.
- **`CORTEX_DEBUG_HTTP` env var** in `core/logging.py`: escape hatch
  symmetric with the existing `CORTEX_DEBUG_MCP_SDK`. When set to any
  non-empty value, leaves the uvicorn / httpx / httpcore loggers at
  their default INFO level instead of silencing them to WARNING. The
  two escape hatches are independent — setting one does NOT re-enable
  the other.

### Fixed — Bundle 10.7

- **B.2 `_process_cmdline` returning None race** (closes the Bundle 8
  handoff hypothesis): when `_build_locked_error` is called with an
  alive holder PID but `_process_cmdline` returns None (ps timeout,
  /proc race, or missing permissions), the code now sets a new
  `cmdline_unknown=True` flag on `StoreLockedError` and refuses to
  auto-recover. The previous code kept `is_pid_reuse=False` silently
  and the error message claimed the live process was the legitimate
  holder — which was unverifiable. `cortex doctor unlock` now reports
  "cmdline could NOT be read — cannot verify this is the same process
  that holds the lock. Refusing to unlock without --force". 4 new
  tests (3 in `test_graph_store_locking.py` for the library path, 1
  in `test_doctor.py` for the CLI path).

### Changed — Bundle 10.7

- **`_quiet_noisy_loggers()`** in `core/logging.py` extended with a
  second silence group `_NOISY_HTTP_LOGGERS` covering `uvicorn`,
  `uvicorn.access`, `httpx`, `httpcore`, `httpcore.http11`, and
  `httpcore.connection`. `uvicorn.error` is intentionally NOT quieted
  so server-side error signal keeps flowing. On a production
  LaunchAgent install these were driving ~33 MB/day of combined log
  growth in `~/.cortex/mcp-http.{log,err}` — dominated by uvicorn's
  per-request access log (one INFO line per MCP HTTP call, from the
  uvicorn that FastMCP starts internally for the streamable-http
  transport) and httpx's outbound INFO lines during sentence-
  transformers embedding model warmup. Empirically verified post-fix:
  `mcp-http.log` is 0 bytes after a burst of CLI traffic (down from
  ~170 lines/minute), and `mcp-http.err` only accumulates the one-time
  sentence-transformers model-load output at server startup.
- **`_MinLevelFilter`** logger-attached filter in `core/logging.py`:
  required because plain `setLevel` on `uvicorn.access` does not stick.
  `uvicorn.Server.configure_logging` unconditionally re-applies the
  config-supplied `log_level` to `uvicorn.access` AFTER `dictConfig`,
  which would overwrite any level we set. A `logging.Filter` on the
  logger instance survives both (a) `dictConfig` with
  `disable_existing_loggers=False` and (b) the explicit `setLevel`,
  so records with `levelno < WARNING` are dropped before reaching
  uvicorn's stream handler.

### Added — Bundle 9 (bug-hunt fixes + CI)

- **GitHub Actions CI** (`.github/workflows/test.yml`): Linux job runs
  ruff + `pytest -n auto` on every push and pull request; macOS job
  runs the full suite including the darwin-only integration tests on
  `main` pushes, the nightly schedule, and manual `workflow_dispatch`.
  Sets `NO_COLOR`, `COLUMNS=200`, and `CORTEX_PROBE_TIMEOUT_SECONDS`
  env vars to stabilize tests on the slower GitHub Actions macOS
  runner. `pytest-xdist` and `pytest-forked` were added to the
  `[dependency-groups].dev` so `uv sync --group dev` installs them.
- **`cortex serve --parent-watchdog`**: hidden flag used by
  `cortex dashboard --spawn-mcp` to make the child MCP server exit
  cleanly when the parent dashboard dies, including on SIGKILL.
  Implemented as a daemon thread in the child that blocks on
  `sys.stdin.read()` and calls `os._exit(0)` when the OS closes the
  pipe. `_spawn_mcp_subprocess` in `cli/main.py` passes `stdin=PIPE`
  and the flag automatically. Closes the A.3 orphan gap discovered
  during the Bundle 9 bug hunt.
- **`CORTEX_PROBE_TIMEOUT_SECONDS` env var**: controls the timeout
  passed to `_get_probe_client()` in `cli/main.py`. Defaults to 10s
  for local use; the CI workflow sets it to 30s because the GitHub
  Actions macOS runner needs more headroom on cold-start tool calls
  that also warm up the embedding model.
- **`_quiet_noisy_loggers()`** in `core/logging.py`: quiets five
  chatty MCP SDK loggers (`mcp.server.lowlevel.server`,
  `mcp.server.streamable_http`, `httpx`, `httpcore.http11`,
  `httpcore.connection`) to WARNING during normal operation. Set
  `CORTEX_DEBUG_MCP_SDK=1` to opt back into the verbose output for
  troubleshooting.

### Changed — Bundle 9

- **`CortexMCPClient` default timeout** bumped from `5.0s` to `10.0s`.
  The 5s default was too narrow against measured p95 of ~1.4s for a
  warm tool call, and flaked under `pytest -n auto` when CPU
  contention pushed real RTTs past the budget.
- **`_get_probe_client()` helper** in `cli/main.py` gives the startup
  probe its own (shorter) timeout budget separate from the tool-call
  client — fast-fail on local use, ample headroom on CI via
  `CORTEX_PROBE_TIMEOUT_SECONDS`.
- **REST API auth hardening** in `verify_api_key`: now uses
  `hmac.compare_digest` to defeat timing-oracle attacks and
  `str.strip()` on the header value so whitespace-padded keys
  compare as equal. A separate re-check rejects keys that become
  empty after strip so whitespace-only keys cannot leak through
  dev mode.
- **`FakePopen[bytes]` subscript workaround**: `tests/cli/test_lock_errors.py`
  pre-imports `cortex.transport.mcp.client` at module load so the
  upstream MCP SDK's `win32/utilities.py` class-definition-time
  subscript runs before any test monkeypatches `subprocess.Popen`.
  Only load-bearing under `pytest --forked`; remove with care.
- **Phase 2 concurrency test isolation**: three test classes
  (`TestConcurrentClients`, `TestMcpHttpServerCrashRecovery`,
  `TestDashboardDoesNotOpenGraphDb`) now carry
  `@pytest.mark.xdist_group("phase2_concurrency")` so xdist
  serializes them within a single worker and avoids CPU contention
  that blew past the CortexMCPClient timeouts.

### Fixed — Bundle 9

- **A.3 dashboard `--spawn-mcp` orphan on SIGKILL**: hard-killing the
  dashboard used to leave the spawned MCP child alive (its `atexit`
  handler never ran), and the orphan kept holding `graph.db.lock`.
  The `--parent-watchdog` flag above closes this by exiting within
  ~1s of the parent pipe closing. Real-subprocess test at
  `tests/cli/test_lock_errors.py::TestParentWatchdog`.

### Added — Bundle 8 (quality-of-life hardening)

- **Auto-recovery of stale locks in `GraphStore.__init__`**: when a lock
  error is hit and the marker file records a PID that is verified dead
  (and not a PID-reuse case), `GraphStore` now removes the stale marker
  and the RocksDB `LOCK` file and retries the open automatically. Users
  no longer need to `rm -rf graph.db/LOCK` by hand after a crash. A
  single `INFO` line ("Auto-recovered stale lock") captures the action.
- **`cortex doctor` subcommand** with `unlock`: guided recovery for
  cases where auto-recovery can't act (e.g. cross-user
  `PermissionError`, or pre-emptive cleanup before starting the
  LaunchAgent). Supports `--dry-run` to inspect and `--force` to bypass
  the live-holder check. Refuses to unlock a running holder by default.
- **`cortex dashboard --spawn-mcp`**: opt-in flag that spawns a
  `cortex serve --transport mcp-http` subprocess when the initial
  probe fails, waits for it to become ready, and terminates the child
  on dashboard exit via an `atexit` handler. Log files at
  `~/.cortex/mcp-http.{log,err}` (same paths the LaunchAgent uses).
- **Shipped LaunchAgent templates**: `deploy/ai.abbacus.cortex.mcp.plist`
  and `deploy/ai.abbacus.cortex.dashboard.plist`, both with `YOURUSER`
  placeholders and install-time `sed` substitution. Covered by a
  `plistlib`-based smoke test in `tests/deploy/test_plist_templates.py`
  that catches XML syntax errors before they land.

### Changed — Bundle 8

- **`StoreLockedError._cleanup_hint`** now points at `cortex doctor
  unlock` as the preferred recovery path, with the raw `rm` commands
  kept as a fallback. When auto-recovery has already been attempted
  and failed, the error message surfaces an explicit
  "Auto-recovery was already attempted and failed" line so users know
  manual intervention is needed.
- **`_raise_locked_error` split into `_build_locked_error` (returns)
  + legacy `_raise_locked_error` (raises)**: the returning variant
  lets `GraphStore.__init__` inspect the candidate error before
  deciding whether to attempt auto-recovery. The raising variant is
  preserved as a thin wrapper for any remaining callers.
- **README.md**: replaced the inline MCP LaunchAgent XML with a
  pointer to `deploy/` and added the dashboard LaunchAgent install +
  startup-race note. New "Recovering from a crashed MCP server"
  section documents auto-recovery and the `doctor unlock` command.
  Removed the stale "REST API still opens graph.db directly"
  limitation — that was resolved by Phase 4.
- **RESUME_INSTRUCTIONS.md**: updated both troubleshooting and
  follow-up sections to use the shipped `deploy/` templates and the
  new `cortex doctor unlock` command.

### Added — earlier phases

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
| After Phase 4 (Bundle 7, REST API as MCP client)    |   796 |
| After Bundle 8 (doctor unlock + spawn-mcp + plists) |   833 |
| After Bundle 9 (bug-hunt fixes + CI)                |   841 |
| After Bundle 10 (SPARQL + templates + ruff)          |   897 |
| After Bundle 10.8 (BaseExceptionGroup)               |   907 |
| After Bundle 10.9 + security probes                  |   927 |
| After Distribution bundle                            |   952 |

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

> **Note**: the previous "REST API still opens graph.db directly"
> limitation is resolved by Phase 4 (Bundle 7). All three transports
> (MCP, dashboard, REST API) plus the CLI now route through the
> canonical MCP HTTP server.
