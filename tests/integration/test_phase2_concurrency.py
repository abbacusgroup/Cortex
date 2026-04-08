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
from pathlib import Path
from typing import Iterator

import pytest

from cortex.transport.mcp.client import (
    CortexMCPClient,
    MCPConnectionError,
)


pytestmark = pytest.mark.slow


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
            )
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
        client = CortexMCPClient(url, timeout_seconds=5.0)
        tools = await client.list_tools()
        assert "cortex_search" in tools
        assert "cortex_capture" in tools
        assert "cortex_status" in tools
        assert "cortex_query_trail" in tools
        assert "cortex_graph_data" in tools

    @pytest.mark.asyncio
    async def test_status_returns_live_data(self, mcp_http_server):
        url, _proc = mcp_http_server
        client = CortexMCPClient(url, timeout_seconds=5.0)
        status = await client.status()
        assert isinstance(status, dict)
        assert "sqlite_total" in status
        assert "graph_triples" in status


class TestConcurrentClients:
    @pytest.mark.asyncio
    async def test_capture_from_one_client_visible_to_another(self, mcp_http_server):
        """The whole point: write via client A, read via client B, no lock fight."""
        url, _proc = mcp_http_server
        client_a = CortexMCPClient(url, timeout_seconds=5.0)
        client_b = CortexMCPClient(url, timeout_seconds=5.0)

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
        client = CortexMCPClient(url, timeout_seconds=5.0)
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
        client = CortexMCPClient(url, timeout_seconds=10.0)

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

        url, _proc = mcp_http_server
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

        # Capture via direct MCP client
        client = CortexMCPClient(url, timeout_seconds=5.0)
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
