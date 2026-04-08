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
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    # Phase 3: lock-error tests verify the direct-store path. Force direct
    # mode so commands like `cortex list` exercise the StoreLockedError flow
    # instead of routing through the live MCP HTTP server.
    monkeypatch.setattr(cli_mod, "_use_mcp", lambda: False)
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False


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
        from cortex.transport.mcp.client import MCPConnectionError

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


class TestApiServerLockError:
    """Bundle 1.4: ``cortex serve --transport http`` (the REST API path)
    must catch StoreLockedError from create_api() and exit cleanly.

    The CLI catches the error before uvicorn starts. There's no live HTTP
    request handling needed because the lock is acquired at startup, not
    per-request.
    """

    def test_create_api_raises_store_locked_error_when_locked(self, held_lock):
        """``create_api()`` should propagate StoreLockedError unchanged."""
        from cortex.core.errors import StoreLockedError
        from cortex.transport.api.server import create_api

        with pytest.raises(StoreLockedError) as exc_info:
            create_api()
        err = exc_info.value
        assert err.holder_pid == held_lock.pid
        assert err.holder_cmdline is not None

    def test_serve_http_cli_exits_cleanly_when_locked(self, held_lock):
        """``cortex serve --transport http`` should exit 1 with a clean
        error when the graph DB is locked, no Python traceback.
        """
        result = runner.invoke(app, ["serve", "--transport", "http"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "locked" in combined.lower()
        assert str(held_lock.pid) in combined
        # No traceback in user-facing output
        assert "Traceback" not in combined


class TestCreateMcpServerLockError:
    """Bundle 4 / A19: Phase 1.D unit-level coverage for create_mcp_server().

    The CLI-level propagation is already covered by
    ``TestServeMcpHttpTransport.test_mcp_http_transport_propagates_lock_error``
    above. This class adds the direct unit test the original A+D plan called
    for: invoking ``create_mcp_server()`` while the graph DB is locked must
    raise ``StoreLockedError`` unchanged, populated with the holder's PID
    and cmdline.
    """

    def test_create_mcp_server_raises_store_locked_error(self, held_lock):
        from cortex.core.errors import StoreLockedError
        from cortex.transport.mcp.server import create_mcp_server

        with pytest.raises(StoreLockedError) as exc_info:
            create_mcp_server()
        err = exc_info.value
        assert err.holder_pid == held_lock.pid
        assert err.holder_cmdline is not None

    def test_create_mcp_server_with_include_admin_false_also_raises(
        self, held_lock
    ):
        """Both the admin-on and admin-off construction paths hit the same
        Store acquisition and should propagate StoreLockedError identically.
        """
        from cortex.core.errors import StoreLockedError
        from cortex.transport.mcp.server import create_mcp_server

        with pytest.raises(StoreLockedError) as exc_info:
            create_mcp_server(include_admin=False)
        err = exc_info.value
        assert err.holder_pid == held_lock.pid


class TestServeMcpHttpPortInUse:
    """Phase 2.A: ``cortex serve --transport mcp-http --port X`` should fail
    cleanly when port X is already in use, NOT crash with a Python traceback.

    The CLI invokes ``run_http`` which calls into uvicorn under the hood;
    uvicorn raises ``OSError: [Errno 48] Address already in use`` on bind
    failure. We mock ``run_http`` to raise that exception and verify the CLI
    handles it gracefully. Bundle 4 tightened this assertion after the CLI
    learned to catch generic OSError.
    """

    def test_mcp_http_port_already_in_use_exits_cleanly(self):
        with patch("cortex.transport.mcp.server.run_http") as mock_run_http:
            mock_run_http.side_effect = OSError(
                "[Errno 48] error while attempting to bind on address ('127.0.0.1', 1314): "
                "address already in use"
            )
            result = runner.invoke(
                app, ["serve", "--transport", "mcp-http", "--port", "1314"]
            )
            assert result.exit_code != 0
            combined = result.output + (result.stderr or "")
            assert "Cannot start MCP HTTP server" in combined
            assert "address already in use" in combined.lower()
            assert "Traceback" not in combined


class TestServeMcpHttpBindErrors:
    """Bundle 4 / A1: non-lock bind failures (PermissionError, generic OSError)
    must produce clean, actionable error messages instead of Python tracebacks.
    """

    def test_mcp_http_privileged_port_exits_cleanly(self):
        """Binding to a privileged port (e.g. 80) raises PermissionError on
        macOS when not root. The CLI must translate that into a clear hint.
        """
        with patch("cortex.transport.mcp.server.run_http") as mock_run_http:
            mock_run_http.side_effect = PermissionError(
                "[Errno 13] Permission denied"
            )
            result = runner.invoke(
                app, ["serve", "--transport", "mcp-http", "--port", "80"]
            )
            assert result.exit_code == 1
            combined = result.output + (result.stderr or "")
            assert "Permission denied" in combined
            # Actionable hint
            assert "1024" in combined or "elevated" in combined.lower()
            assert "Traceback" not in combined

    def test_mcp_http_generic_oserror_exits_cleanly(self):
        """Other OSErrors (not lock, not permission) also exit cleanly —
        e.g. ``[Errno 49] Can't assign requested address`` for a bad host.
        """
        with patch("cortex.transport.mcp.server.run_http") as mock_run_http:
            mock_run_http.side_effect = OSError(
                "[Errno 49] Can't assign requested address"
            )
            result = runner.invoke(
                app,
                [
                    "serve",
                    "--transport",
                    "mcp-http",
                    "--host",
                    "192.0.2.1",
                    "--port",
                    "1314",
                ],
            )
            assert result.exit_code == 1
            combined = result.output + (result.stderr or "")
            assert "Cannot start MCP HTTP server" in combined
            assert "192.0.2.1" in combined
            assert "1314" in combined
            assert "Traceback" not in combined
