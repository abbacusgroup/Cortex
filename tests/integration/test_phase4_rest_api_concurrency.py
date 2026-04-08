"""End-to-end concurrency tests for Phase 4 (REST API as MCP HTTP client).

Proves that ``cortex serve --transport http`` (the REST API) can run
simultaneously with ``cortex serve --transport mcp-http`` (the MCP HTTP
server). Before Phase 4 both entry points opened ``graph.db`` directly,
which triggered the Phase 1 honest-mode lock error. After Phase 4 the
REST API is a thin HTTP client of the MCP server, so they coexist.

Marked ``@pytest.mark.slow`` because it spawns real subprocesses.
"""

from __future__ import annotations

import json as _json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="macOS-only: uses ps, lsof, POSIX signals, and launchd semantics",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── Helpers ──────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, *, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except (urllib.error.URLError, OSError) as e:
            last_error = e
            time.sleep(0.2)
    raise TimeoutError(
        f"HTTP endpoint {url} never became ready: {last_error}"
    )


def _spawn_mcp_http_server(
    tmp_path: Path, *, port: int | None = None
) -> tuple[str, subprocess.Popen]:
    if port is None:
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
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait until the MCP HTTP endpoint is up by polling a known admin
    # tool via the REST-API-independent ``CortexMCPClient`` wrapper.
    import asyncio
    import threading

    from cortex.transport.mcp.client import CortexMCPClient

    deadline = time.time() + 15
    ready = [False]
    last_error: list[Exception | None] = [None]

    def _try_once() -> None:
        client = CortexMCPClient(url, timeout_seconds=2.0)
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(client.list_tools())
                ready[0] = True
            finally:
                loop.close()
        except Exception as e:
            last_error[0] = e

    while time.time() < deadline:
        t = threading.Thread(target=_try_once)
        t.start()
        t.join(timeout=5)
        if ready[0]:
            return url, proc
        time.sleep(0.2)

    proc.terminate()
    proc.wait(timeout=10)
    stdout = proc.stdout.read() if proc.stdout else ""
    stderr = proc.stderr.read() if proc.stderr else ""
    raise TimeoutError(
        f"MCP HTTP server at {url} failed to start: {last_error[0]}\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    )


def _spawn_rest_api(
    tmp_path: Path, mcp_url: str, *, port: int | None = None
) -> tuple[str, subprocess.Popen]:
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
            "serve",
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{port}/health")
    except TimeoutError:
        proc.terminate()
        proc.wait(timeout=10)
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        raise TimeoutError(
            f"REST API at port {port} failed to start.\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    return f"http://127.0.0.1:{port}", proc


@pytest.fixture
def mcp_plus_rest_api(
    tmp_path: Path,
) -> Iterator[tuple[str, str, subprocess.Popen, subprocess.Popen]]:
    """Spawn both the MCP HTTP server AND the REST API pointing at it."""
    mcp_url, mcp_proc = _spawn_mcp_http_server(tmp_path)
    rest_proc: subprocess.Popen | None = None
    try:
        rest_url, rest_proc = _spawn_rest_api(tmp_path, mcp_url)
        yield mcp_url, rest_url, mcp_proc, rest_proc
    finally:
        if rest_proc is not None:
            rest_proc.terminate()
            try:
                rest_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rest_proc.kill()
                rest_proc.wait()
        if mcp_proc.poll() is None:
            mcp_proc.terminate()
            try:
                mcp_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                mcp_proc.kill()
                mcp_proc.wait()


# ─── Tests ────────────────────────────────────────────────────────────────


class TestRestApiAndMcpHttpCoexist:
    """Phase 4 contract: the REST API and the MCP HTTP server can run
    simultaneously. Before Phase 4 this failed with the Phase 1 lock
    error because both opened graph.db directly.
    """

    def test_both_servers_start_successfully(self, mcp_plus_rest_api):
        mcp_url, rest_url, mcp_proc, rest_proc = mcp_plus_rest_api
        # Both alive
        assert mcp_proc.poll() is None
        assert rest_proc.poll() is None
        # REST /health responds
        with urllib.request.urlopen(
            f"{rest_url}/health", timeout=5
        ) as resp:
            assert resp.status == 200

    def test_rest_api_search_forwards_to_mcp(self, mcp_plus_rest_api):
        mcp_url, rest_url, _mcp, _rest = mcp_plus_rest_api
        req = urllib.request.Request(
            f"{rest_url}/search?query=test",
            method="POST",
            headers={"X-API-Key": "dev"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            body = _json.loads(resp.read())
            assert isinstance(body, list)

    def test_rest_api_capture_visible_via_mcp(self, mcp_plus_rest_api):
        """Capture via REST API → read back via the same MCP server
        through a direct MCP client call. Proves both entry points
        see the same store.
        """
        import asyncio

        from cortex.transport.mcp.client import CortexMCPClient

        mcp_url, rest_url, _mcp, _rest = mcp_plus_rest_api

        # Capture via REST API
        req = urllib.request.Request(
            f"{rest_url}/capture?title=Phase4%20test&content=x&obj_type=idea&run_pipeline=false",
            method="POST",
            headers={"X-API-Key": "dev"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            body = _json.loads(resp.read())
            new_id = body["id"]

        # Read back via direct MCP client
        client = CortexMCPClient(mcp_url, timeout_seconds=30.0)

        async def _read():
            return await client.read(new_id)

        result = asyncio.run(_read())
        assert isinstance(result, dict)
        assert result["title"] == "Phase4 test"

    def test_only_mcp_server_holds_graph_db(
        self, mcp_plus_rest_api, tmp_path: Path
    ):
        """lsof verification: the REST API PID must NOT appear among the
        processes holding files under ``graph.db``. Only the MCP server
        should. This is the Phase 4 contract made concrete.
        """
        _mcp_url, rest_url, mcp_proc, rest_proc = mcp_plus_rest_api
        graph_db = tmp_path / "graph.db"
        if not graph_db.exists():
            pytest.skip(f"graph.db not found at {graph_db}")

        # Drive some traffic through the REST API so it definitely
        # exercises its MCP client path.
        req = urllib.request.Request(
            f"{rest_url}/status", headers={"X-API-Key": "dev"}
        )
        urllib.request.urlopen(req, timeout=5).read()

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
            f"MCP server PID {mcp_proc.pid} not among graph.db holders "
            f"{holder_pids}; lsof stdout:\n{result.stdout}"
        )
        assert rest_proc.pid not in holder_pids, (
            f"REST API PID {rest_proc.pid} unexpectedly holds files "
            f"under {graph_db}! Phase 4 contract violated. "
            f"Holders: {holder_pids}"
        )


class TestRestApiStartupProbe:
    """Phase 4: ``cortex serve --transport http`` probes the MCP server
    at startup and fails fast if it's unreachable.
    """

    def test_rest_api_fails_fast_when_mcp_unreachable(
        self, tmp_path: Path
    ):
        """With no MCP HTTP server running, the REST API CLI should exit
        1 with an actionable error pointing at
        ``cortex serve --transport mcp-http``.
        """
        port = _free_port()
        mcp_url = "http://127.0.0.1:1/mcp"  # guaranteed-closed port
        env = os.environ.copy()
        env["CORTEX_DATA_DIR"] = str(tmp_path)
        env["CORTEX_MCP_SERVER_URL"] = mcp_url
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "cortex.cli.main",
                "serve",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail(
                "REST API did not exit within 15s despite MCP unreachable"
            )

        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        combined = stdout + stderr

        assert rc != 0, (
            f"expected non-zero exit; got rc={rc}, combined={combined!r}"
        )
        assert "Cannot reach Cortex MCP server" in combined or "MCP" in combined
        # Actionable hint
        assert "cortex serve --transport mcp-http" in combined
        assert "Traceback" not in combined
