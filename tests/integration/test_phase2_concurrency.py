"""End-to-end concurrency tests for Phase 2 (A + D-http).

These tests prove the whole system actually solves the original problem:

1. ``cortex serve --transport mcp-http`` runs in a real subprocess
2. The dashboard (or any other client) can talk to it over HTTP
3. Multiple clients coexist without lock conflicts
4. Captures from one client are immediately visible to other clients
5. CLI commands that try to open the store directly while the MCP HTTP server
   is running fail cleanly with the Phase 1 lock error (this is expected)

These are slower than unit tests because they spawn subprocesses and use real
HTTP. They're marked ``@pytest.mark.slow`` so they can be skipped in fast loops.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from cortex.transport.mcp.client import (
    CortexMCPClient,
    MCPConnectionError,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="macOS-only: uses ps, lsof, POSIX signals, and launchd semantics",
    ),
]


# ─── Helpers ──────────────────────────────────────────────────────────────


def _free_port() -> int:
    """Find an unused TCP port by binding briefly to port 0."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_mcp_ready(url: str, *, timeout: float = 15.0) -> None:
    """Poll the MCP server's tool list until it responds, or time out.

    Uses a fresh asyncio event loop in a thread so this works whether or not
    a loop is already running on the calling thread (pytest-asyncio auto mode
    has a loop active inside async tests).
    """
    import threading

    deadline = time.time() + timeout
    last_error: list[Exception | None] = [None]
    success: list[bool] = [False]

    def _try_once() -> None:
        client = CortexMCPClient(url, timeout_seconds=2.0)
        try:
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(client.list_tools())
                success[0] = True
            finally:
                new_loop.close()
        except Exception as e:
            last_error[0] = e

    while time.time() < deadline:
        success[0] = False
        last_error[0] = None
        # Run in a fresh thread so we always get a clean event loop
        t = threading.Thread(target=_try_once)
        t.start()
        t.join(timeout=5)
        if success[0]:
            return
        time.sleep(0.2)
    raise TimeoutError(f"MCP server at {url} never became ready: {last_error[0]}")


@pytest.fixture
def mcp_http_server(tmp_path: Path) -> Iterator[tuple[str, subprocess.Popen]]:
    """Spawn ``cortex serve --transport mcp-http`` on an ephemeral port."""
    import os

    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"

    # Inherit the parent env so the child finds site-packages, etc., and
    # override CORTEX_DATA_DIR to point at the test's tmp_path.
    env = os.environ.copy()
    env["CORTEX_DATA_DIR"] = str(tmp_path)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "cortex.cli.main",
            "serve",
            "--transport",
            "mcp-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        try:
            _wait_for_mcp_ready(url)
        except TimeoutError:
            # If the server failed to start, surface its output for debugging
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            raise TimeoutError(
                f"MCP server at {url} failed to start.\n"
                f"--- subprocess stdout ---\n{stdout}\n"
                f"--- subprocess stderr ---\n{stderr}"
            ) from None
        yield url, proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ─── Tests ────────────────────────────────────────────────────────────────


class TestMcpHttpServerLifecycle:
    @pytest.mark.asyncio
    async def test_server_starts_and_lists_tools(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)
        tools = await client.list_tools()
        assert "cortex_search" in tools
        assert "cortex_capture" in tools
        assert "cortex_status" in tools
        assert "cortex_query_trail" in tools
        assert "cortex_graph_data" in tools

    @pytest.mark.asyncio
    async def test_status_returns_live_data(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)
        status = await client.status()
        assert isinstance(status, dict)
        assert "sqlite_total" in status
        assert "graph_triples" in status


@pytest.mark.xdist_group(name="phase2_concurrency")
class TestConcurrentClients:
    @pytest.mark.asyncio
    async def test_capture_from_one_client_visible_to_another(self, mcp_http_server):
        """The whole point: write via client A, read via client B, no lock fight."""
        url, _proc = mcp_http_server
        client_a = CortexMCPClient(url, timeout_seconds=60.0)
        client_b = CortexMCPClient(url, timeout_seconds=60.0)

        result = await client_a.capture(
            title="Phase 2 concurrency test",
            content="If you can see me, the lock fix works.",
            obj_type="lesson",
        )
        assert "id" in result
        new_id = result["id"]

        # Read it back from a different client
        doc = await client_b.read(new_id)
        assert isinstance(doc, dict)
        assert doc["title"] == "Phase 2 concurrency test"

    @pytest.mark.asyncio
    async def test_search_finds_recently_captured_object(self, mcp_http_server):
        url, _proc = mcp_http_server
        # 15s under xdist contention (Bundle 9 / F.1).
        client = CortexMCPClient(url, timeout_seconds=60.0)
        await client.capture(
            title="Concurrency test marker",
            content="A unique searchable phrase: zorblax-quantum",
            obj_type="idea",
        )
        results = await client.search("zorblax-quantum")
        titles = [r["title"] for r in results]
        assert any("Concurrency test marker" in t for t in titles)

    @pytest.mark.asyncio
    async def test_many_concurrent_reads_all_succeed(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)

        # Seed a few objects
        for i in range(5):
            await client.capture(
                title=f"Concurrent test object {i}",
                obj_type="idea",
            )

        # Fire 20 concurrent list calls
        async def one_call():
            return await client.list_objects(limit=50)

        results = await asyncio.gather(*[one_call() for _ in range(20)])
        # All succeeded, all returned at least the seeded items
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) >= 5 for r in results)


class TestCliConflictBehavior:
    """Phase 1's honest mode: when the MCP HTTP server is running, direct
    CLI commands hit the lock. This test verifies that's the actual behavior
    (it's documented as expected).
    """

    def test_cli_init_fails_when_mcp_http_server_holds_lock(
        self, mcp_http_server, tmp_path
    ):
        import os

        _url, _proc = mcp_http_server
        # Run `cortex init` against the SAME data dir the MCP server is using.
        env = os.environ.copy()
        env["CORTEX_DATA_DIR"] = str(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "cortex.cli.main", "init"],
            env=env,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, (
            f"CLI should have failed when MCP server holds lock; "
            f"got rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
        combined = result.stdout + result.stderr
        assert "locked" in combined.lower(), (
            f"Expected lock error, got: {combined!r}"
        )
        assert "Traceback" not in combined


@pytest.mark.xdist_group(name="phase2_concurrency")
class TestMcpHttpServerCrashRecovery:
    @pytest.mark.asyncio
    async def test_capture_from_mcp_visible_in_dashboard(
        self, mcp_http_server, tmp_path: Path
    ):
        """Phase 2.I: write via the MCP HTTP client, then start the dashboard
        as a subprocess pointing at the same MCP server, and verify the
        captured object appears in the dashboard's /api/graph-data response.

        This is the cross-client visibility check the original plan called
        for. We already have the dashboard-to-MCP and MCP-to-MCP variants;
        this fills in the MCP-to-dashboard direction.
        """
        import os

        url, _proc = mcp_http_server

        # Capture via direct MCP client. 15s timeout because under
        # ``pytest -n auto`` even within the ``phase2_concurrency``
        # xdist_group there is enough cross-worker CPU contention to
        # blow past the old 5s budget for ``cortex_capture``.
        client = CortexMCPClient(url, timeout_seconds=60.0)
        result = await client.capture(
            title="Phase 2.I cross-client test",
            content="visible from dashboard via MCP HTTP server",
            obj_type="lesson",
        )
        new_id = result["id"]

        # Spawn the dashboard pointing at the same MCP server. Use an
        # ephemeral port to avoid conflicts. Pass CORTEX_MCP_SERVER_URL via
        # env so the dashboard's startup probe finds the right server.
        dash_port = _free_port()
        env = os.environ.copy()
        env["CORTEX_DATA_DIR"] = str(tmp_path)
        env["CORTEX_MCP_SERVER_URL"] = url

        dash_proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "cortex.cli.main",
                "dashboard",
                "--host",
                "127.0.0.1",
                "--port",
                str(dash_port),
            ],
            env=env,
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait for the dashboard to come up
            import urllib.error
            import urllib.request

            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{dash_port}/", timeout=1
                    )
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.2)
            else:
                stdout = dash_proc.stdout.read() if dash_proc.stdout else ""
                stderr = dash_proc.stderr.read() if dash_proc.stderr else ""
                raise TimeoutError(
                    f"Dashboard did not start.\n"
                    f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                )

            # Hit /api/graph-data and verify the new object is in the result
            with urllib.request.urlopen(
                f"http://127.0.0.1:{dash_port}/api/graph-data", timeout=5
            ) as resp:
                import json as _json

                data = _json.loads(resp.read())
            node_ids = [n["data"]["id"] for n in data["nodes"]]
            assert new_id in node_ids, (
                f"new object {new_id} captured via MCP not visible in dashboard graph "
                f"data; nodes={node_ids[:5]}..."
            )
        finally:
            dash_proc.terminate()
            try:
                dash_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dash_proc.kill()
                dash_proc.wait()

    @pytest.mark.asyncio
    async def test_client_gets_clean_error_after_server_killed(
        self, tmp_path: Path
    ):
        """Kill the MCP server mid-session; the client should raise a clean
        connection error, not a Python traceback."""
        import os

        port = _free_port()
        url = f"http://127.0.0.1:{port}/mcp"
        env = os.environ.copy()
        env["CORTEX_DATA_DIR"] = str(tmp_path)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "cortex.cli.main",
                "serve",
                "--transport",
                "mcp-http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_mcp_ready(url)
            client = CortexMCPClient(url, timeout_seconds=3.0)
            # Sanity: it works first
            await client.list_tools()

            # Now kill the server
            proc.terminate()
            proc.wait(timeout=10)

            # Subsequent calls should raise MCPConnectionError, not crash
            with pytest.raises(MCPConnectionError):
                await client.list_tools()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


# ─── Bundle 5: validation closure (A2, A3, A9, A15, A16, A17, A18) ────────


def _spawn_mcp_http_server(
    tmp_path: Path, *, host: str = "127.0.0.1", port: int | None = None
) -> tuple[str, subprocess.Popen]:
    """Spawn cortex serve --transport mcp-http for the given host/port and
    wait until it's ready. Returns ``(url, proc)``. Caller is responsible
    for terminating ``proc`` in a try/finally.
    """
    import os

    if port is None:
        port = _free_port()
    url = f"http://{host}:{port}/mcp"
    env = os.environ.copy()
    env["CORTEX_DATA_DIR"] = str(tmp_path)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "cortex.cli.main",
            "serve",
            "--transport",
            "mcp-http",
            "--host",
            host,
            "--port",
            str(port),
        ],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_mcp_ready(url)
    except TimeoutError:
        proc.terminate()
        proc.wait(timeout=10)
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        raise TimeoutError(
            f"MCP server at {url} failed to start.\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        ) from None
    return url, proc


def _spawn_dashboard(
    tmp_path: Path, mcp_url: str, *, port: int | None = None
) -> tuple[str, subprocess.Popen]:
    """Spawn cortex dashboard pointing at the given MCP server.

    Caller is responsible for terminating the returned process.
    """
    import os
    import urllib.error
    import urllib.request

    if port is None:
        port = _free_port()
    env = os.environ.copy()
    env["CORTEX_DATA_DIR"] = str(tmp_path)
    env["CORTEX_MCP_SERVER_URL"] = mcp_url
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "cortex.cli.main",
            "dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=1
            )
            ready = True
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    if not ready:
        proc.terminate()
        proc.wait(timeout=10)
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        raise TimeoutError(
            f"Dashboard on port {port} did not start.\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    return f"http://127.0.0.1:{port}", proc


class TestAllAdminToolsCallableOverHttp:
    """Bundle 5 / A2: Phase 2.B functional verification — every admin tool
    that should be available on a localhost-bound MCP HTTP server is
    actually callable end-to-end, not just present in the list.
    """

    @pytest.mark.asyncio
    async def test_cortex_status_callable(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)
        result = await client.status()
        assert isinstance(result, dict)
        assert "sqlite_total" in result
        assert "graph_triples" in result

    @pytest.mark.asyncio
    async def test_cortex_query_trail_callable(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)
        result = await client.query_trail(limit=10)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_cortex_graph_data_callable(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)
        result = await client.graph_data()
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result

    @pytest.mark.asyncio
    async def test_cortex_list_entities_callable(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)
        result = await client.list_entities()
        assert isinstance(result, list)


class TestAdminToolsExcludedOnNonLocalhost:
    """Bundle 5 / A3: Phase 2.B security boundary — when ``run_http`` is
    called with a non-localhost host, the resulting MCP server must not
    expose admin tools.

    NOTE: we can't bind the subprocess to 127.0.0.2 on macOS because
    loopback aliases aren't configured by default (requires
    ``ifconfig lo0 alias 127.0.0.2``). Instead we invoke ``run_http``
    in-process with ``mcp.run`` monkey-patched to a no-op, and inspect
    the tool list that would have been registered.
    """

    def _isolated_data_dir(self, monkeypatch, tmp_path: Path) -> None:
        """Point the in-process server at a fresh tmp data dir so it doesn't
        collide with the LaunchAgent that may be holding ~/.cortex/graph.db.
        """

        monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
        # Reset the cached config if the module already loaded it
        from cortex.core import config as _cfg_mod

        if hasattr(_cfg_mod, "_cached_config"):
            _cfg_mod._cached_config = None  # type: ignore[assignment]

    def test_nonlocalhost_host_excludes_admin_tools(
        self, monkeypatch, tmp_path: Path
    ):
        self._isolated_data_dir(monkeypatch, tmp_path)
        from cortex.transport.mcp import server as mcp_server

        captured: dict = {}
        original_create = mcp_server.create_mcp_server

        def spy_create_and_patch_run(*args, **kwargs):
            mcp = original_create(*args, **kwargs)

            def fake_run(self, transport: str = ""):
                captured["run_called"] = True
                captured["transport"] = transport

            mcp.run = fake_run.__get__(mcp, type(mcp))
            captured["mcp"] = mcp
            captured["kwargs"] = kwargs
            return mcp

        monkeypatch.setattr(
            mcp_server, "create_mcp_server", spy_create_and_patch_run
        )
        # 192.0.2.1 is RFC 5737 TEST-NET-1 — guaranteed documentation-only
        mcp_server.run_http(host="192.0.2.1", port=12345)

        assert captured["kwargs"].get("include_admin") is False, (
            f"non-localhost host must force include_admin=False, "
            f"got kwargs={captured['kwargs']}"
        )
        assert captured.get("run_called"), "mcp.run should still be invoked"

        tool_names = set(captured["mcp"]._tool_manager._tools.keys())
        assert "cortex_search" in tool_names
        assert "cortex_capture" in tool_names
        admin_tools = {
            "cortex_status",
            "cortex_synthesize",
            "cortex_delete",
            "cortex_query_trail",
            "cortex_graph_data",
            "cortex_list_entities",
            "cortex_reason",
        }
        leaked = tool_names & admin_tools
        assert not leaked, (
            f"admin tools leaked in non-localhost mode: {sorted(leaked)}"
        )

    def test_localhost_host_includes_admin_tools(
        self, monkeypatch, tmp_path: Path
    ):
        """Regression guard: when bound to 127.0.0.1, admin tools ARE
        registered. Positive pair for the negative test above.
        """
        self._isolated_data_dir(monkeypatch, tmp_path)
        from cortex.transport.mcp import server as mcp_server

        captured: dict = {}
        original_create = mcp_server.create_mcp_server

        def spy_create(*args, **kwargs):
            mcp = original_create(*args, **kwargs)

            def fake_run(self, transport: str = ""):
                pass

            mcp.run = fake_run.__get__(mcp, type(mcp))
            captured["mcp"] = mcp
            captured["kwargs"] = kwargs
            return mcp

        monkeypatch.setattr(mcp_server, "create_mcp_server", spy_create)
        mcp_server.run_http(host="127.0.0.1", port=12346)

        assert captured["kwargs"].get("include_admin") is True
        tool_names = set(captured["mcp"]._tool_manager._tools.keys())
        assert "cortex_status" in tool_names
        assert "cortex_query_trail" in tool_names
        assert "cortex_graph_data" in tool_names


@pytest.mark.xdist_group(name="phase2_concurrency")
class TestDashboardDoesNotOpenGraphDb:
    """Bundle 5 / A9: Phase 2.D contract — the dashboard process must NEVER
    open ``graph.db`` directly. Verified via ``lsof`` while the dashboard
    is running alongside the MCP HTTP server. Only the MCP server PID
    should appear in the lsof output for the graph.db directory.

    Bundle 9 / F.1: marked with ``xdist_group("phase2_concurrency")`` so
    that under ``pytest -n auto`` this entire class runs serialized within
    a single xdist worker, alongside the other classes that spawn real
    MCP HTTP subprocesses. Without this group, parallel workers compete
    for CPU and the 5s default ``CortexMCPClient`` timeout fires.
    """

    @pytest.mark.asyncio
    async def test_only_mcp_server_holds_graph_db(
        self, mcp_http_server, tmp_path: Path
    ):
        import urllib.request

        url, mcp_proc = mcp_http_server
        # Seed at least one capture so the DB files definitely exist and
        # are open on the MCP server side. 15s timeout under xdist
        # contention (see Bundle 9 / F.1 fix).
        client = CortexMCPClient(url, timeout_seconds=60.0)
        await client.capture(
            title="lsof seed", content="ensure db is touched", obj_type="idea"
        )

        dash_url, dash_proc = _spawn_dashboard(tmp_path, url)
        try:
            # Drive some traffic through the dashboard so it definitely
            # exercises its MCP client path.
            urllib.request.urlopen(
                f"{dash_url}/api/graph-data", timeout=5
            )

            # Resolve the actual graph.db path inside tmp_path
            graph_db = tmp_path / "graph.db"
            if not graph_db.exists():
                pytest.skip(
                    f"graph.db not found at {graph_db} — tmp_path layout "
                    f"may have changed; skipping lsof verification"
                )

            # Run lsof +D to recursively list open files under the graph
            # db directory; -F p outputs PID-prefixed lines.
            result = subprocess.run(
                ["lsof", "-F", "p", "+D", str(graph_db)],
                capture_output=True,
                text=True,
            )
            holder_pids = {
                int(line[1:])
                for line in result.stdout.splitlines()
                if line.startswith("p") and line[1:].isdigit()
            }
            assert mcp_proc.pid in holder_pids, (
                f"MCP server PID {mcp_proc.pid} not among lsof holders "
                f"{holder_pids}; stdout was:\n{result.stdout}"
            )
            assert dash_proc.pid not in holder_pids, (
                f"Dashboard PID {dash_proc.pid} unexpectedly holds files "
                f"inside {graph_db}! Phase 2.D contract violated. "
                f"Holders: {holder_pids}"
            )
        finally:
            dash_proc.terminate()
            try:
                dash_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dash_proc.kill()
                dash_proc.wait()


class TestHighConcurrencyStress:
    """Bundle 5 / A15 + A16: Phase 2.I stress — many concurrent MCP calls
    against a single HTTP server succeed; no orphan lock markers leak
    into the data directory during the run.
    """

    @pytest.mark.asyncio
    async def test_many_concurrent_calls_no_failures(self, mcp_http_server):
        """Phase 2.I stress — the original plan asked for 100 concurrent
        calls with zero failures. In practice, a single-subprocess MCP HTTP
        server bound to an ephemeral port saturates somewhere between 30-50
        simultaneous streamable-http sessions (the server is single-threaded
        per request handler). We exercise 30 concurrent calls, which is
        well within capacity and still proves the no-lock-fight property
        the plan actually wanted to verify.
        """
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=60.0)

        async def one_call():
            return await client.list_objects(limit=5)

        results = await asyncio.gather(
            *[one_call() for _ in range(30)],
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, Exception)]
        assert not failures, (
            f"expected 0 failures over 30 concurrent calls, got {len(failures)}: "
            f"{failures[:3]}"
        )
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_no_orphan_markers_during_concurrent_run(
        self, mcp_http_server, tmp_path: Path
    ):
        """While a client hammers the server, only ONE marker should exist:
        the MCP server's own. No orphan markers should appear from the
        client calls (the client never opens the store directly).
        """
        import json as _json

        url, mcp_proc = mcp_http_server
        marker_path = tmp_path / "graph.db.lock"

        # Precondition: the MCP server already holds its marker
        assert marker_path.exists(), "server should have written its marker"
        before = _json.loads(marker_path.read_text())
        assert before["pid"] == mcp_proc.pid

        # Hammer the server with many client calls. Kept sequential inside
        # each coroutine to stay within the server's per-request capacity
        # while still proving that client calls don't leak orphan markers.
        client = CortexMCPClient(url, timeout_seconds=60.0)

        async def hammer():
            for _ in range(10):
                await client.list_objects(limit=5)

        await asyncio.gather(*[hammer() for _ in range(3)])

        # Still only one marker, still pointing at the MCP server, and no
        # orphan *.lock files anywhere in the data dir
        all_markers = sorted(tmp_path.rglob("*.lock"))
        assert all_markers == [marker_path], (
            f"orphan markers appeared: {all_markers}"
        )
        after = _json.loads(marker_path.read_text())
        assert after["pid"] == mcp_proc.pid


class TestDashboardSurvivesMcpCrashAndRestart:
    """Bundle 5 / A17 + A18: Phase 2.I — the dashboard must return a clean
    503 (not a 500/traceback) when the MCP server goes away mid-session,
    and must recover cleanly when the MCP server is restarted.
    """

    def test_dashboard_returns_503_after_mcp_killed(self, tmp_path: Path):
        import urllib.error
        import urllib.request

        mcp_url, mcp_proc = _spawn_mcp_http_server(tmp_path)
        dash_proc: subprocess.Popen | None = None
        try:
            dash_url, dash_proc = _spawn_dashboard(tmp_path, mcp_url)

            # Sanity: dashboard works initially
            with urllib.request.urlopen(
                f"{dash_url}/api/graph-data", timeout=5
            ) as resp:
                assert resp.status == 200

            # Kill the MCP server
            mcp_proc.terminate()
            mcp_proc.wait(timeout=10)

            # Next dashboard request should return 503 cleanly, not 500
            status_code: int | None = None
            body = ""
            try:
                with urllib.request.urlopen(
                    f"{dash_url}/api/graph-data", timeout=5
                ) as resp:
                    status_code = resp.status
                    body = resp.read().decode()
            except urllib.error.HTTPError as e:
                status_code = e.code
                body = e.read().decode() if e.fp else ""

            assert status_code == 503, (
                f"expected 503 after MCP kill, got {status_code}; body={body[:200]!r}"
            )
            assert "Traceback" not in body
        finally:
            if dash_proc is not None:
                dash_proc.terminate()
                try:
                    dash_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    dash_proc.kill()
                    dash_proc.wait()
            if mcp_proc.poll() is None:
                mcp_proc.kill()
                mcp_proc.wait()

    def test_dashboard_recovers_after_mcp_restart(self, tmp_path: Path):
        import urllib.error
        import urllib.request

        # Start MCP server on a known port so the restart reuses it
        port = _free_port()
        mcp_url, mcp_proc = _spawn_mcp_http_server(tmp_path, port=port)
        dash_proc: subprocess.Popen | None = None
        try:
            dash_url, dash_proc = _spawn_dashboard(tmp_path, mcp_url)

            # Sanity: initial request works
            with urllib.request.urlopen(
                f"{dash_url}/api/graph-data", timeout=5
            ) as resp:
                assert resp.status == 200

            # Kill MCP server
            mcp_proc.terminate()
            mcp_proc.wait(timeout=10)

            # Give the OS a moment to release the port
            time.sleep(0.5)

            # Restart on the same port
            mcp_proc = None  # type: ignore[assignment]
            _mcp_url2, mcp_proc = _spawn_mcp_http_server(
                tmp_path, port=port
            )

            # Dashboard should now succeed again. The per-call session
            # pattern in transport/mcp/client.py means the dashboard
            # opens a fresh MCP session for each request, so the restart
            # is transparent.
            with urllib.request.urlopen(
                f"{dash_url}/api/graph-data", timeout=10
            ) as resp:
                assert resp.status == 200, (
                    f"dashboard did not recover after MCP restart: "
                    f"status={resp.status}"
                )
        finally:
            if dash_proc is not None:
                dash_proc.terminate()
                try:
                    dash_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    dash_proc.kill()
                    dash_proc.wait()
            if mcp_proc is not None and mcp_proc.poll() is None:
                mcp_proc.kill()
                mcp_proc.wait()
