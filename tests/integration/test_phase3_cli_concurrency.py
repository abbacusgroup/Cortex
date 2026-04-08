"""End-to-end concurrency tests for Phase 3.

Proves that CLI commands route through a real MCP HTTP server when one is
running, and that ``--direct`` correctly bypasses the routing (and hits the
Phase 1 lock error if the server holds the lock).

Marked ``@pytest.mark.slow`` because it spawns subprocesses.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from cortex.transport.mcp.client import CortexMCPClient

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="macOS-only: uses ps, lsof, POSIX signals, and launchd semantics",
    ),
]


# ─── Helpers ──────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_mcp_ready(url: str, *, timeout: float = 15.0) -> None:
    """Poll the MCP server's tool list until ready, or time out.

    Uses a fresh asyncio loop in a thread to avoid event-loop conflicts.
    """
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
        t = threading.Thread(target=_try_once)
        t.start()
        t.join(timeout=5)
        if success[0]:
            return
        time.sleep(0.2)
    raise TimeoutError(f"MCP server at {url} never became ready: {last_error[0]}")


@pytest.fixture
def mcp_http_server(tmp_path: Path) -> Iterator[tuple[str, subprocess.Popen]]:
    """Spawn ``cortex serve --transport mcp-http`` on an ephemeral port + isolated data dir."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"

    env = os.environ.copy()
    env["CORTEX_DATA_DIR"] = str(tmp_path)
    env["CORTEX_MCP_SERVER_URL"] = url  # CLI subprocesses point at this URL

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
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            raise TimeoutError(
                f"MCP server at {url} failed to start.\n"
                f"--- subprocess stdout ---\n{stdout}\n"
                f"--- subprocess stderr ---\n{stderr}"
            )
        yield url, proc, env, tmp_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _run_cli(env: dict, args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run `python -m cortex.cli.main <args>` with the given env."""
    return subprocess.run(
        [sys.executable, "-u", "-m", "cortex.cli.main", *args],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ─── Tests ────────────────────────────────────────────────────────────────


class TestCliRoutesThroughMcpServer:
    def test_cortex_status_via_mcp(self, mcp_http_server):
        url, _proc, env, _tmp = mcp_http_server
        result = _run_cli(env, ["status"])
        assert result.returncode == 0, (
            f"cortex status failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        # Should reflect the EMPTY tmp_path data dir, not the user's real data
        assert "Documents:" in result.stdout
        assert "Triples:" in result.stdout

    def test_cortex_list_via_mcp(self, mcp_http_server):
        url, _proc, env, _tmp = mcp_http_server
        result = _run_cli(env, ["list"])
        assert result.returncode == 0
        # Empty store: should print "No objects found."
        assert "No objects" in result.stdout or "object(s)" in result.stdout

    def test_capture_then_read_then_list_round_trip(self, mcp_http_server):
        """Capture an object via MCP, then read it back, then list — all via MCP."""
        url, _proc, env, _tmp = mcp_http_server

        # Capture
        cap = _run_cli(
            env,
            [
                "capture",
                "Phase 3 e2e test object",
                "--type",
                "idea",
                "--content",
                "captured via cortex CLI through MCP HTTP",
            ],
        )
        assert cap.returncode == 0, f"capture failed: {cap.stderr}"
        # Extract the object ID from the output
        lines = cap.stdout.strip().split("\n")
        first_line = lines[0]  # "Captured idea: <uuid>"
        obj_id = first_line.split(": ")[-1].strip()
        assert len(obj_id) > 8

        # Read it back
        read = _run_cli(env, ["read", obj_id])
        assert read.returncode == 0
        assert "Phase 3 e2e test object" in read.stdout

        # List should include it
        lst = _run_cli(env, ["list"])
        assert lst.returncode == 0
        assert "Phase 3 e2e test" in lst.stdout

    def test_search_finds_captured_object(self, mcp_http_server):
        url, _proc, env, _tmp = mcp_http_server
        _run_cli(
            env,
            [
                "capture",
                "uniqueze4ke marker idea",
                "--type",
                "idea",
                "--content",
                "uniqueze4ke marker text body",
            ],
        )
        result = _run_cli(env, ["search", "uniqueze4ke"])
        assert result.returncode == 0
        assert "uniqueze4ke" in result.stdout


class TestDirectFlagInteraction:
    def test_direct_with_running_mcp_hits_lock_error(self, mcp_http_server):
        """While the MCP HTTP server holds the lock, --direct should fail
        with the Phase 1 lock error. This documents that --direct bypasses
        MCP routing but NOT the actual graph DB lock.
        """
        url, _proc, env, _tmp = mcp_http_server
        result = _run_cli(env, ["--direct", "list"])
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "locked" in combined.lower()
        # The MCP HTTP server is the holder, so its cmdline should appear
        assert "mcp-http" in combined or "cortex serve" in combined
        # No traceback in user-facing output
        assert "Traceback" not in combined


class TestNoMcpNoDirect:
    def test_actionable_error_when_mcp_unreachable_and_no_direct(self, tmp_path):
        """Without an MCP server running AND without --direct, the CLI should
        exit with a clear error mentioning BOTH options (start server OR --direct).
        """
        env = os.environ.copy()
        env["CORTEX_DATA_DIR"] = str(tmp_path)
        # Point at a closed port — connection will be refused
        env["CORTEX_MCP_SERVER_URL"] = "http://127.0.0.1:1/mcp"

        result = _run_cli(env, ["list"])
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "Cannot reach" in combined or "MCP server" in combined
        # Mentions both recovery options
        assert "cortex serve --transport mcp-http" in combined
        assert "--direct" in combined
        # No traceback
        assert "Traceback" not in combined


class TestConcurrentCliCalls:
    def test_many_concurrent_cli_invocations(self, mcp_http_server):
        """Fire 10 concurrent `cortex list` invocations against the same MCP
        server. All should succeed; no lock errors, no port issues.
        """
        url, _proc, env, _tmp = mcp_http_server

        results: list[subprocess.CompletedProcess] = []
        threads: list[threading.Thread] = []

        def _runner():
            results.append(_run_cli(env, ["list"]))

        for _ in range(10):
            t = threading.Thread(target=_runner)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        for r in results:
            assert r.returncode == 0, f"concurrent CLI failed: {r.stderr}"
