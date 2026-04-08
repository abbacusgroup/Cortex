# Resuming the Cortex A+D-http production switch

This file is the bridge from the 2026-04-07 implementation session to your next Claude Code session. Read it once, do the manual steps in order, then paste the prompt at the end into your next conversation.

---

## What was done in the previous session

Phase 1 (honest single-writer mode) and Phase 2 (D-http: MCP server as canonical writer) were implemented, tested, and committed locally. **673/673 tests passing** (was 549 before).

**Key code changes** (already on disk in `~/Lab/cortex`, NOT yet committed to git):

| File | What changed |
|---|---|
| `src/cortex/core/errors.py` | Added `StoreLockedError` |
| `src/cortex/db/graph_store.py` | PID marker write/read, raises `StoreLockedError` on lock conflict, `close()` releases the RocksDB lock |
| `src/cortex/db/store.py` | `Store.close()` now also closes the graph store |
| `src/cortex/core/config.py` | Added `mcp_server_url` field with HTTP/HTTPS validation |
| `src/cortex/cli/main.py` | `_open_store_or_exit()` helper, wrapped CLI entry points, added `--transport mcp-http`, dashboard startup probe with required-tools check, `if __name__ == "__main__"` block |
| `src/cortex/transport/mcp/server.py` | `run_http()` uses `mcp.settings.host/port` (not kwargs), localhost-bound `run_http` enables admin tools, added `cortex_query_trail` / `cortex_graph_data` / `cortex_list_entities` admin tools |
| `src/cortex/dashboard/server.py` | **Rewritten** as thin MCP HTTP client. No more Store/PipelineOrchestrator/RetrievalEngine/LearningLoop/LLMClient imports. |
| `src/cortex/dashboard/mcp_client.py` | **NEW**: ~250 LOC async wrapper around `mcp.client.streamable_http` |
| `src/cortex/dashboard/templates/error.html` | **NEW**: friendly error page when MCP unreachable |
| `pyproject.toml` | Registered `slow` pytest marker |

**New tests** (~124 total): `tests/db/test_graph_store_locking.py`, `tests/cli/test_lock_errors.py`, `tests/dashboard/test_mcp_client.py`, `tests/dashboard/test_server_mcp_mode.py`, `tests/integration/test_phase2_concurrency.py`, plus extensions to existing test files.

**Memory checkpoint:** `/Users/fabrizzio/.claude/projects/-Users-fabrizzio-Lab-cortex/memory/project_oxigraph_lock_fix.md` — automatically loaded by future Claude sessions in `~/Lab/cortex`.

**Plan reference:** `/Users/fabrizzio/.claude/plans/binary-singing-llama.md` — full design doc with validation plan and weak-points table.

---

## What was changed manually at the end of the session

1. **`~/.claude.json`** — Cortex MCP entry switched from stdio to HTTP transport
2. **Backup** — original config saved as `~/.claude.json.bak.20260407`

The new entry looks like this:

```json
"cortex": {
  "type": "http",
  "url": "http://127.0.0.1:1314/mcp"
}
```

Until you start a `cortex serve --transport mcp-http` process and restart Claude Code, **Cortex MCP tools will not be available**. The next steps fix that.

---

## Steps to do RIGHT NOW (in this exact order)

These steps are interdependent. Don't skip ahead.

### Step 1 — Quit Claude Code

Quit the running Claude Code app (Cmd+Q). This:
- Releases the lock on `~/.cortex/graph.db` (Claude Code's stdio MCP child process exits)
- Saves this conversation to disk so you can resume it later

### Step 2 — Open a long-lived terminal and start the MCP HTTP server

In a fresh terminal that you're going to **leave running**:

```bash
cd ~/Lab/cortex
.venv/bin/cortex serve --transport mcp-http --host 127.0.0.1 --port 1314
```

Expected output:
```
Cortex MCP (streamable-http) at http://127.0.0.1:1314/mcp
INFO:     Started server process [...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:1314 (Press CTRL+C to quit)
```

Keep this terminal open. Don't Ctrl+C it. You can put it in a tmux pane, a screen session, or just minimize the window.

If you want this to start automatically on login eventually, wrap it in a launchd plist — that's a separate job for later, not required right now.

### Step 3 — Verify the MCP server is reachable

In a different terminal:

```bash
curl -s http://127.0.0.1:1314/mcp -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke-test","version":"1"}}}' \
  | head -3
```

You should get back a JSON response (or SSE event). HTTP 200. If you get connection refused, the server in step 2 isn't running.

### Step 4 — Start Claude Code again

Open Claude Code normally. It will read `~/.claude.json`, see the new HTTP transport for Cortex, and connect to `http://127.0.0.1:1314/mcp`.

### Step 5 — Resume this conversation

In Claude Code:

```
claude --resume
```

Pick the conversation titled something like "Continue current session" or with the most recent timestamp from today (2026-04-07). The full history of this implementation session will be loaded.

### Step 6 — Paste this prompt into the resumed session

Once you're back in the resumed conversation, paste this as your first message:

```
We just completed the production switch for the Cortex Oxigraph A+D-http
lock fix. Phase 1 + Phase 2 are implemented, ~/.claude.json was edited to
use the HTTP MCP transport, and I just restarted Claude Code. Can you:

1. Verify Cortex MCP tools are connected by calling cortex_status — you
   should see ~160 docs and ~10458 graph triples (the real production data
   from the 2026-04-06 import)
2. Confirm the dashboard works by starting `cortex dashboard --port 1315`
   in a background terminal and curling http://127.0.0.1:1315/
3. Run the full test suite from ~/Lab/cortex to confirm 673 tests still
   pass: `.venv/bin/python -m pytest tests/ -q`
4. Check that the demo environment from earlier is fully cleaned up (no
   leftover processes on ports 18800/18801, no /tmp/cortex-demo dir)
5. If everything looks good, ask me whether I want to commit the Phase 1+2
   work to git as a single commit, or split it into Phase 1 + Phase 2
   commits

The full design and validation plan is at:
/Users/fabrizzio/.claude/plans/binary-singing-llama.md

The handoff memory is at:
/Users/fabrizzio/.claude/projects/-Users-fabrizzio-Lab-cortex/memory/project_oxigraph_lock_fix.md
(loaded automatically when working in ~/Lab/cortex)
```

---

## If something goes wrong

### The MCP HTTP server fails to start (lock conflict)

If you see `Graph DB at ~/.cortex/graph.db is locked by another process` when running `cortex serve --transport mcp-http`, it means another process still holds the lock. Most likely:
- An old Claude Code session is still running somewhere — quit all Claude Code instances
- A leftover stdio MCP process from before the restart — find and kill it: `lsof ~/.cortex/graph.db | grep -v COMMAND | awk '{print $2}' | sort -u | xargs kill`

Then retry step 2.

### Claude Code says "MCP server not connected" or shows Cortex as disconnected

This usually means Claude Code is running BEFORE the HTTP server is started. Just leave Claude Code open and run step 2 in a separate terminal. Claude Code should reconnect within a few seconds.

If it doesn't reconnect automatically, restart Claude Code one more time after the HTTP server is confirmed running.

### Cortex tools throw errors when called from Claude Code

Run `cortex_status` to see what's wrong. Most likely the MCP HTTP server logs (in your long-lived terminal from step 2) will show the actual error. Common issues:
- Wrong data dir: confirm `CORTEX_DATA_DIR` env var if you set one
- Missing ontology: run `cd ~/Lab/cortex && .venv/bin/cortex init` once to make sure
- Missing tools: confirm the running server is from `~/Lab/cortex/.venv/bin/cortex` (the version with the new tools), not some old install

### You want to roll back to the old stdio config

```bash
cp ~/.claude.json.bak.20260407 ~/.claude.json
```

Restart Claude Code. This restores the pre-2026-04-07 stdio MCP entry. The implementation work in `~/Lab/cortex` is unaffected by this — only the Claude Code config changes.

### Dashboard `/create` fails

The dashboard talks to MCP via HTTP. If MCP is down, the dashboard returns a 503. If MCP is up but `cortex_capture` fails, the dashboard returns a 502 with the error body. Check the MCP server's terminal log for the actual exception.

---

## One-time eventual followup: LaunchAgent for the MCP HTTP server

You currently start the MCP HTTP server manually in a terminal (step 2). Eventually you'll want it to start automatically on login. The launchd plist looks roughly like this — save to `~/Library/LaunchAgents/ai.abbacus.cortex.mcp.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.abbacus.cortex.mcp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/fabrizzio/Lab/cortex/.venv/bin/cortex</string>
        <string>serve</string>
        <string>--transport</string>
        <string>mcp-http</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>1314</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/fabrizzio/.cortex/mcp-http.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/fabrizzio/.cortex/mcp-http.err</string>
    <key>WorkingDirectory</key>
    <string>/Users/fabrizzio/Lab/cortex</string>
</dict>
</plist>
```

Then `launchctl load ~/Library/LaunchAgents/ai.abbacus.cortex.mcp.plist`. After that the MCP HTTP server starts automatically every time you log in. Don't bother with this unless you want it — manual `cortex serve` in a tmux pane works fine.

---

## Claude Code MCP session staleness (upstream issue)

After you install the LaunchAgent (or any time you restart the MCP HTTP server
manually), Claude Code may start failing its Cortex MCP tool calls with an
error like:

```
{"code":-32600,"message":"Session not found"}
```

This is because Claude Code's MCP client establishes a **session ID** the
first time it connects to the HTTP MCP server. When the server restarts
(launchctl reload, `KeepAlive` auto-restart, manual Ctrl+C and relaunch), the
new server process doesn't recognize the old session ID and rejects
subsequent calls.

**This is an upstream Claude Code behavior, not a Cortex bug.** The fix is to
restart Claude Code itself:

1. `Cmd+Q` to quit Claude Code
2. Re-open Claude Code
3. `claude --resume` to pick up the conversation you were in

**Workarounds that do NOT work:**
- `claude /mcp reconnect` — does not refresh the session ID
- Waiting for KeepAlive to settle — the server is healthy, Claude Code just
  has a stale handle
- Restarting only the LaunchAgent — Claude Code still holds the old session

**The dashboard and CLI do NOT have this issue.** `transport/mcp/client.py`
opens a fresh MCP session per request (per-call session pattern), so the
dashboard and `cortex` CLI commands reconnect transparently after any MCP
restart. Only long-lived MCP client connections (Claude Code) are affected.

**In practice**, you only hit this when you intentionally restart the MCP
server. Under KeepAlive the server only restarts if it crashes, which is
rare. Still, if Cortex tools suddenly stop working from Claude Code, the
first thing to try is `Cmd+Q` and re-open.
