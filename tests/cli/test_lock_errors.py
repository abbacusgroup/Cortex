"""Tests for the CLI's clean handling of StoreLockedError.

Verifies Phase 1.D: every CLI entry point that opens the Store catches
StoreLockedError, prints a user-friendly red message, and exits with code 1.
Also covers Phase 2.A: the new ``--transport mcp-http`` CLI option.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import cortex.cli.main as cli_mod
from cortex.cli.main import app
from cortex.db.graph_store import _marker_path_for

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None


@pytest.fixture
def held_lock(tmp_path):
    """Spawn a subprocess that holds the graph DB lock for the duration of the test."""
    db_path = tmp_path / "graph.db"
    sentinel = tmp_path / "ready"
    code = (
        "import sys, time; sys.path.insert(0, 'src');"
        "from pathlib import Path;"
        "from cortex.db.graph_store import GraphStore;"
        f"s = GraphStore(Path({str(db_path)!r}));"
        f"Path({str(sentinel)!r}).write_text('ready');"
        "time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
    )
    # Wait for the holder to acquire the lock
    deadline = time.time() + 10
    while time.time() < deadline:
        if sentinel.exists():
            break
        time.sleep(0.05)
    else:
        proc.terminate()
        proc.wait()
        pytest.fail("Holder subprocess did not acquire the lock in time")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class TestCliLockErrors:
    def test_init_exits_cleanly_when_locked(self, held_lock):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "locked" in result.output.lower() or "locked" in (result.stderr or "").lower()
        # No Python traceback in user-facing output
        combined = result.output + (result.stderr or "")
        assert "Traceback" not in combined
        assert "StoreLockedError" not in combined

    def test_init_error_message_includes_holder_pid(self, held_lock):
        result = runner.invoke(app, ["init"])
        combined = result.output + (result.stderr or "")
        assert str(held_lock.pid) in combined, (
            f"Expected holder PID {held_lock.pid} in CLI output, got: {combined!r}"
        )

    def test_setup_exits_cleanly_when_locked(self, held_lock):
        result = runner.invoke(app, ["setup", "--auto"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "locked" in combined.lower()
        assert "Traceback" not in combined

    def test_get_store_exits_cleanly_when_locked(self, held_lock):
        # `cortex list` goes through _get_store(); when locked, should exit cleanly
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "locked" in combined.lower()
        assert "Traceback" not in combined

    def test_does_NOT_silently_succeed_with_empty_data(self, held_lock):
        """Regression: the OLD code's silent in-memory fallback would let
        `cortex list` return zero results as if everything was fine. New code
        must fail loudly instead.
        """
        result = runner.invoke(app, ["list"])
        # The exit code MUST be non-zero. A zero exit with empty output
        # would mean the bug is back.
        assert result.exit_code != 0


class TestMarkerLifecycleViaCli:
    def test_init_creates_then_removes_marker(self, tmp_path):
        # CORTEX_DATA_DIR is already set to tmp_path by the autouse fixture
        marker = _marker_path_for(tmp_path / "graph.db")
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        # After CliRunner exits, the store is closed via the singleton's __del__
        # path or atexit. The CliRunner runs in-process, so the singleton
        # _store is still held. Force-cleanup:
        if cli_mod._store is not None:
            cli_mod._store.graph.close()
        assert not marker.exists(), (
            f"Marker file should be removed after store close, found: {marker}"
        )


class TestServeMcpHttpTransport:
    """Phase 2.A: ``cortex serve --transport mcp-http`` wires up run_http.

    The ``run_http`` function blocks indefinitely, so we mock it.
    """

    def test_mcp_http_transport_calls_run_http(self):
        with patch("cortex.transport.mcp.server.run_http") as mock_run_http:
            result = runner.invoke(
                app, ["serve", "--transport", "mcp-http", "--port", "1314"]
            )
            assert result.exit_code == 0
            mock_run_http.assert_called_once_with(host="127.0.0.1", port=1314)

    def test_mcp_http_default_port_is_1314(self):
        with patch("cortex.transport.mcp.server.run_http") as mock_run_http:
            result = runner.invoke(app, ["serve", "--transport", "mcp-http"])
            assert result.exit_code == 0
            kwargs = mock_run_http.call_args.kwargs
            assert kwargs["port"] == 1314

    def test_mcp_http_default_host_is_localhost(self):
        with patch("cortex.transport.mcp.server.run_http") as mock_run_http:
            result = runner.invoke(app, ["serve", "--transport", "mcp-http"])
            assert result.exit_code == 0
            kwargs = mock_run_http.call_args.kwargs
            assert kwargs["host"] == "127.0.0.1"

    def test_mcp_http_custom_host_and_port(self):
        with patch("cortex.transport.mcp.server.run_http") as mock_run_http:
            result = runner.invoke(
                app,
                ["serve", "--transport", "mcp-http", "--host", "0.0.0.0", "--port", "9999"],
            )
            assert result.exit_code == 0
            mock_run_http.assert_called_once_with(host="0.0.0.0", port=9999)

    def test_stdio_transport_unchanged(self):
        """Regression: stdio transport still works after adding mcp-http."""
        with patch("cortex.transport.mcp.server.run_stdio") as mock_run_stdio:
            result = runner.invoke(app, ["serve", "--transport", "stdio"])
            assert result.exit_code == 0
            mock_run_stdio.assert_called_once()

    def test_http_transport_still_runs_rest_api(self):
        """Regression: --transport http still starts the REST API, NOT the MCP HTTP server."""
        with patch("cortex.transport.api.server.create_api") as mock_create_api, \
             patch("uvicorn.run") as mock_uvicorn_run:
            mock_create_api.return_value = "fake_api"
            result = runner.invoke(app, ["serve", "--transport", "http"])
            assert result.exit_code == 0
            mock_create_api.assert_called_once()
            mock_uvicorn_run.assert_called_once_with(
                "fake_api", host="127.0.0.1", port=1314
            )

    def test_unknown_transport_rejected(self):
        result = runner.invoke(app, ["serve", "--transport", "gibberish"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "Unknown transport" in combined or "gibberish" in combined
        # Error mentions valid options
        assert "stdio" in combined and "mcp-http" in combined

    def test_mcp_http_transport_propagates_lock_error(self, held_lock):
        """When the graph DB is locked, mcp-http should exit cleanly with the lock error."""
        result = runner.invoke(app, ["serve", "--transport", "mcp-http"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "locked" in combined.lower()
        assert "Traceback" not in combined


class TestDashboardStartupProbe:
    """Phase 2.F: ``cortex dashboard`` probes the MCP server before starting."""

    def test_dashboard_starts_when_mcp_reachable(self, monkeypatch):
        """Happy path: probe succeeds, list_tools returns the expected set."""
        # Mock both _probe_mcp_server and uvicorn.run
        import cortex.cli.main as main_mod

        monkeypatch.setattr(
            main_mod,
            "_probe_mcp_server",
            lambda url, **kw: main_mod._REQUIRED_MCP_TOOLS,
        )
        with patch("uvicorn.run") as mock_uvicorn, patch(
            "cortex.dashboard.server.create_dashboard"
        ) as mock_create:
            mock_create.return_value = "fake_app"
            result = runner.invoke(app, ["dashboard", "--port", "1315"])
            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()

    def test_dashboard_fails_when_mcp_unreachable(self, monkeypatch):
        """When the MCP server probe fails, dashboard exits cleanly."""
        from cortex.dashboard.mcp_client import MCPConnectionError

        def fake_probe(url, **kw):
            raise MCPConnectionError(
                f"Cannot reach MCP server at {url}",
                holder_pid=None if False else None,  # not actually a lock error
            )

        # MCPConnectionError doesn't take holder_pid kwarg — fix
        def real_fake_probe(url, **kw):
            raise MCPConnectionError(f"Cannot reach MCP server at {url}")

        monkeypatch.setattr("cortex.cli.main._probe_mcp_server", real_fake_probe)
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "Cannot reach" in combined or "MCP server" in combined
        # Helpful error includes the start command
        assert "cortex serve --transport mcp-http" in combined
        assert "Traceback" not in combined

    def test_dashboard_fails_when_mcp_missing_required_tools(self, monkeypatch):
        """If the MCP server is missing tools the dashboard needs, exit cleanly."""
        # Probe returns only 1 tool — missing all the required ones
        monkeypatch.setattr(
            "cortex.cli.main._probe_mcp_server",
            lambda url, **kw: {"cortex_search"},  # missing the rest
        )
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "missing" in combined.lower()
        assert "cortex_capture" in combined or "cortex_list_entities" in combined
