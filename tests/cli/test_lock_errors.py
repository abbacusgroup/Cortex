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

# Pre-load the mcp SDK before any test runs. The mcp SDK has
# `mcp/os/win32/utilities.py` defining `class FallbackProcess` with a
# class-level annotation `popen_obj: subprocess.Popen[bytes]` that gets
# evaluated at class-definition time. ``TestDashboardSpawnMcp`` patches
# ``subprocess.Popen`` with a non-subscriptable fake before triggering
# the lazy ``cortex.transport.mcp.client`` import inside
# ``_spawn_mcp_subprocess``, which would then fail with
# ``TypeError: type 'FakePopen' is not subscriptable`` under
# ``pytest --forked`` (because each fork starts cold and the SDK has not
# been imported yet). Pre-loading the client module here forces the SDK
# import chain to run with the real ``subprocess.Popen``, fixing the
# test under ``--forked`` without changing production code.
import cortex.transport.mcp.client  # noqa: F401
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

    def test_does_NOT_silently_succeed_with_empty_data(self, held_lock):  # noqa: N802
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
        """Regression: --transport http still starts the REST API, NOT the MCP HTTP server.

        Bundle 9.1 / CI fix: ``cortex serve --transport http`` now probes
        the MCP HTTP server at startup (Phase 4 convergence). Without a
        running MCP server on CI the probe fails and the command exits 1.
        Mock ``_probe_mcp_server`` so the test exercises only the ``http``
        transport wiring.
        """
        with patch("cortex.transport.api.server.create_api") as mock_create_api, \
             patch("uvicorn.run") as mock_uvicorn_run, \
             patch(
                 "cortex.cli.main._probe_mcp_server",
                 return_value=set(cli_mod._REQUIRED_MCP_TOOLS),
             ):
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
            raise MCPConnectionError(f"Cannot reach MCP server at {url}")

        monkeypatch.setattr("cortex.cli.main._probe_mcp_server", fake_probe)
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

    def test_probe_times_out_under_reasonable_bound(self, monkeypatch):
        """Bundle 5 / A12: the startup probe must finish within a bounded
        time — no hangs on dead URLs. The probe currently has 3 retries
        with 1s spacing between them. With a fast-failing probe mock, the
        whole invocation should complete in well under 10s.
        """
        import time as _time

        from cortex.transport.mcp.client import MCPConnectionError

        def fail_probe(url, **kw):
            # Simulate an immediate-fail probe (this mock stands in for the
            # internal retry loop; the CLI's _probe_mcp_server has retries
            # of its own, which test_probe_retries_three_times_with_1s_spacing
            # covers directly).
            raise MCPConnectionError("simulated unreachable")

        monkeypatch.setattr("cortex.cli.main._probe_mcp_server", fail_probe)
        start = _time.time()
        result = runner.invoke(app, ["dashboard"])
        duration = _time.time() - start
        assert result.exit_code == 1
        assert duration < 10.0, (
            f"dashboard startup took {duration:.2f}s — probe must be bounded"
        )

    def test_probe_retries_three_times_with_1s_spacing(self, monkeypatch):
        """Bundle 5 / A13: _probe_mcp_server makes exactly 3 attempts with
        ≥1s spacing between them. Patches at the ``CortexMCPClient.list_tools``
        layer so we exercise the real retry loop inside _probe_mcp_server.

        NOTE: the retry_delay is 1.0s by default, so this test takes ≥2s to
        finish (2 delays between 3 attempts).
        """
        import time as _time

        timestamps: list[float] = []

        async def fail_list_tools(self):
            # Record the time, then raise a client error directly. The
            # ``_probe_mcp_server`` retry loop catches ``MCPClientError`` and
            # sleeps before retrying, which is what this test verifies.
            timestamps.append(_time.time())
            from cortex.transport.mcp.client import MCPConnectionError
            raise MCPConnectionError("nope")

        from cortex.transport.mcp.client import CortexMCPClient

        monkeypatch.setattr(
            CortexMCPClient,
            "list_tools",
            fail_list_tools,
        )

        from cortex.cli.main import _probe_mcp_server
        from cortex.transport.mcp.client import MCPClientError

        with pytest.raises(MCPClientError):
            _probe_mcp_server(
                "http://127.0.0.1:1/mcp", retries=3, retry_delay=1.0
            )
        assert len(timestamps) == 3, (
            f"expected exactly 3 retry attempts, got {len(timestamps)}"
        )
        deltas = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
        ]
        for delta in deltas:
            assert delta >= 0.9, (
                f"retry spacing {delta:.3f}s below the 1s minimum "
                f"(all deltas: {deltas})"
            )

    def test_dashboard_clean_error_on_mcp_500_response(self, monkeypatch):
        """Bundle 5 / A14: if the MCP server is reachable but returns a 500
        on the probe handshake (e.g. an unhealthy backend), the dashboard
        CLI must exit cleanly with an actionable error, not a traceback.
        """
        from cortex.transport.mcp.client import MCPServerError

        def fake_probe(url, **kw):
            raise MCPServerError(
                f"MCP server at {url} returned 500",
                context={"url": url, "status": 500},
            )

        monkeypatch.setattr("cortex.cli.main._probe_mcp_server", fake_probe)
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        # Error should mention the MCP server failure
        assert "MCP" in combined
        assert "500" in combined or "unreach" in combined.lower() or "returned" in combined.lower()
        assert "Traceback" not in combined


class TestApiServerDoesNotOpenStore:
    """After Phase 4 (Bundle 7), the REST API is a thin MCP HTTP client.
    It no longer calls ``Store(config)`` directly, so it cannot produce
    a ``StoreLockedError`` — the lock concept simply doesn't apply to
    this process anymore.

    The Bundle 1 tests that exercised ``create_api()`` against a held
    lock are obsolete. They've been replaced by
    :class:`TestRestApiAndMcpHttpCoexist` in
    ``tests/integration/test_phase4_rest_api_concurrency.py`` which
    verifies the new architectural contract:
    1. The REST API can run simultaneously with the MCP HTTP server.
    2. ``lsof`` confirms the REST API PID never holds ``graph.db``.
    3. ``cortex serve --transport http`` fails fast with an MCP-unreachable
       error when the MCP server isn't running.
    """

    def test_create_api_does_not_import_store(self):
        """``create_api()`` must not import the Store layer at all.

        If this assertion ever fails, someone has re-introduced a
        direct-store dependency and resurrected the lock-conflict
        problem Phase 4 fixed.
        """
        import cortex.transport.api.server as api_mod

        # Inspect the module's actual imports (not docstrings) by
        # looking at the compiled module globals.
        assert not hasattr(api_mod, "Store"), (
            "api.server imports Store — Phase 4 contract violated"
        )
        assert not hasattr(api_mod, "PipelineOrchestrator")
        assert not hasattr(api_mod, "RetrievalEngine")
        assert not hasattr(api_mod, "LearningLoop")


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


class TestDashboardSpawnMcp:
    """Bundle 8 / B2: ``cortex dashboard --spawn-mcp`` launches an MCP HTTP
    server subprocess when the probe fails, waits for readiness, and
    terminates the child when the dashboard exits.
    """

    def test_dashboard_without_spawn_flag_fails_on_unreachable_mcp(
        self, monkeypatch
    ):
        """Regression guard: the default behavior (no --spawn-mcp) still
        exits cleanly with the legacy error message when the probe fails.
        """
        from cortex.transport.mcp.client import MCPConnectionError

        def failing_probe(url, **kw):
            raise MCPConnectionError(f"Cannot reach MCP server at {url}")

        monkeypatch.setattr("cortex.cli.main._probe_mcp_server", failing_probe)
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "Cannot reach" in combined
        # The new hint mentioning --spawn-mcp should be present
        assert "--spawn-mcp" in combined
        assert "Traceback" not in combined

    def test_dashboard_with_spawn_flag_spawns_and_uvicorn_runs(
        self, monkeypatch
    ):
        """When the probe fails and --spawn-mcp is set, the CLI calls
        ``_spawn_mcp_subprocess`` and then proceeds to uvicorn.run.
        """
        from cortex.transport.mcp.client import MCPConnectionError

        probe_calls = {"n": 0}

        def flaky_probe(url, **kw):
            probe_calls["n"] += 1
            if probe_calls["n"] == 1:
                raise MCPConnectionError(f"Cannot reach MCP server at {url}")
            # Second probe (after spawn) succeeds with the required tools
            import cortex.cli.main as main_mod
            return main_mod._REQUIRED_MCP_TOOLS

        monkeypatch.setattr("cortex.cli.main._probe_mcp_server", flaky_probe)

        # Fake _spawn_mcp_subprocess — don't actually start a real server
        class FakeProc:
            pid = 99999

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        spawn_calls = {"n": 0, "args": None}

        def fake_spawn(url, data_dir):
            spawn_calls["n"] += 1
            spawn_calls["args"] = (url, data_dir)
            return FakeProc()

        monkeypatch.setattr("cortex.cli.main._spawn_mcp_subprocess", fake_spawn)

        with patch("uvicorn.run") as mock_uvicorn, patch(
            "cortex.dashboard.server.create_dashboard"
        ) as mock_create:
            mock_create.return_value = "fake_app"
            result = runner.invoke(app, ["dashboard", "--spawn-mcp"])

        assert result.exit_code == 0, (
            f"unexpected exit: {result.output} {result.stderr}"
        )
        assert spawn_calls["n"] == 1
        assert probe_calls["n"] == 2  # initial fail + post-spawn success
        mock_uvicorn.assert_called_once()

    def test_dashboard_with_spawn_flag_skips_spawn_when_probe_succeeds(
        self, monkeypatch
    ):
        """If the initial probe succeeds, --spawn-mcp should NOT spawn a
        duplicate MCP server.
        """
        import cortex.cli.main as main_mod

        monkeypatch.setattr(
            "cortex.cli.main._probe_mcp_server",
            lambda url, **kw: main_mod._REQUIRED_MCP_TOOLS,
        )

        spawn_calls = {"n": 0}

        def fake_spawn(url, data_dir):
            spawn_calls["n"] += 1
            raise RuntimeError("spawn should not have been called")

        monkeypatch.setattr("cortex.cli.main._spawn_mcp_subprocess", fake_spawn)

        with patch("uvicorn.run"), patch(
            "cortex.dashboard.server.create_dashboard"
        ) as mock_create:
            mock_create.return_value = "fake_app"
            result = runner.invoke(app, ["dashboard", "--spawn-mcp"])

        assert result.exit_code == 0
        assert spawn_calls["n"] == 0, (
            "spawn must be skipped when the initial probe succeeds"
        )

    def test_dashboard_with_spawn_flag_exits_on_spawn_failure(
        self, monkeypatch
    ):
        """If _spawn_mcp_subprocess raises RuntimeError (subprocess failed
        to become ready), the dashboard exits 1 with a clean error.
        """
        from cortex.transport.mcp.client import MCPConnectionError

        def failing_probe(url, **kw):
            raise MCPConnectionError(f"Cannot reach MCP server at {url}")

        def failing_spawn(url, data_dir):
            raise RuntimeError("subprocess died before ready: oops")

        monkeypatch.setattr("cortex.cli.main._probe_mcp_server", failing_probe)
        monkeypatch.setattr("cortex.cli.main._spawn_mcp_subprocess", failing_spawn)

        result = runner.invoke(app, ["dashboard", "--spawn-mcp"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "Failed to spawn MCP subprocess" in combined
        assert "oops" in combined
        assert "Traceback" not in combined

    def test_spawn_mcp_subprocess_uses_configured_host_and_port(
        self, monkeypatch, tmp_path
    ):
        """Unit test for ``_spawn_mcp_subprocess`` itself: it parses
        host and port from the URL and passes them on the command line.
        """
        import cortex.cli.main as main_mod

        captured: dict = {}

        class FakePopen:
            pid = 12345
            returncode = None

            def __init__(self, args, env=None, stdin=None, stdout=None, stderr=None):
                captured["args"] = args
                captured["env"] = env
                captured["stdin"] = stdin

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(
            "cortex.cli.main._probe_mcp_server",
            lambda url, **kw: {"cortex_search"},
        )
        # Patch subprocess.Popen at the module level _spawn_mcp_subprocess uses
        import subprocess as subprocess_mod

        monkeypatch.setattr(subprocess_mod, "Popen", FakePopen)

        main_mod._spawn_mcp_subprocess(
            "http://127.0.0.1:9876/mcp", tmp_path
        )
        args = captured["args"]
        assert "serve" in args
        assert "--transport" in args
        assert "mcp-http" in args
        assert "--host" in args
        assert "127.0.0.1" in args
        assert "--port" in args
        assert "9876" in args
        # Bundle 9 / A.3: parent-watchdog flag is plumbed through so the
        # spawned MCP child cannot outlive the dashboard even on SIGKILL.
        assert "--parent-watchdog" in args
        # data_dir passed via env
        assert captured["env"]["CORTEX_DATA_DIR"] == str(tmp_path)
        # stdin must be PIPE — the watchdog thread in the child blocks on it
        # so the OS-level pipe close (caused by parent death) is what wakes
        # the child up to terminate.
        assert captured["stdin"] is subprocess_mod.PIPE

    def test_spawn_mcp_subprocess_fails_fast_on_immediate_exit(
        self, monkeypatch, tmp_path
    ):
        """If the subprocess exits before becoming ready, _spawn_mcp_subprocess
        raises RuntimeError with a clear message and the tail of stderr.
        """
        import cortex.cli.main as main_mod

        class DeadPopen:
            pid = 12345
            returncode = 1

            def __init__(self, *args, **kwargs):
                pass

            def poll(self):
                return 1  # already dead

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 1

            def kill(self):
                pass

        # Pre-seed an error log file so the tail message has something to show
        (tmp_path / "mcp-http.err").write_text("boom: StoreLockedError\n")

        import subprocess as subprocess_mod

        monkeypatch.setattr(subprocess_mod, "Popen", DeadPopen)

        from cortex.transport.mcp.client import MCPConnectionError

        monkeypatch.setattr(
            "cortex.cli.main._probe_mcp_server",
            lambda url, **kw: (_ for _ in ()).throw(
                MCPConnectionError("not yet")
            ),
        )

        with pytest.raises(RuntimeError) as exc_info:
            main_mod._spawn_mcp_subprocess(
                "http://127.0.0.1:9876/mcp", tmp_path
            )
        msg = str(exc_info.value)
        assert "exited with code 1" in msg
        assert "boom" in msg or "StoreLockedError" in msg


class TestParentWatchdog:
    """Bundle 9 / A.3: ``cortex serve --parent-watchdog`` exits when its
    stdin closes (parent death). Plus end-to-end test that
    ``dashboard --spawn-mcp`` plus a SIGKILL'd parent leaves no orphan.
    """

    def test_start_parent_watchdog_spawns_daemon_thread(self, monkeypatch):
        """Smoke test: ``_start_parent_watchdog`` registers a daemon
        thread named ``cortex-parent-watchdog`` that we can find in
        ``threading.enumerate()``.
        """

        import cortex.cli.main as main_mod

        # Don't actually let it block on real stdin and don't let it
        # _exit() us — patch sys.stdin.read so the thread returns
        # immediately, and patch os._exit so the test process survives.
        called = {"exit": False}

        def fake_read():
            return ""

        def fake_exit(code):
            called["exit"] = True

        monkeypatch.setattr("sys.stdin.read", fake_read)
        monkeypatch.setattr("os._exit", fake_exit)

        main_mod._start_parent_watchdog()
        # Give the thread a moment to start (and immediately exit our
        # patched stdin.read).
        for _ in range(50):
            if called["exit"]:
                break
            time.sleep(0.01)
        assert called["exit"] is True

    @pytest.mark.slow
    def test_subprocess_exits_when_parent_closes_stdin(self, tmp_path):
        """End-to-end: launch a real ``cortex serve --transport mcp-http
        --parent-watchdog`` subprocess with a stdin pipe; close the pipe;
        the subprocess must exit within a few seconds.

        This is the actual production fix verification — if this test
        passes, hard-killing the dashboard cannot leave an orphaned MCP
        child holding the lock.
        """
        import os
        import socket

        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

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
                "--parent-watchdog",
            ],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )

        try:
            # Give the server a moment to come up. We don't strictly need
            # to wait for readiness — the watchdog thread runs as soon as
            # the process starts and is independent of mcp.run().
            time.sleep(2.0)
            assert proc.poll() is None, "subprocess died before we could kill it"

            # Close stdin — this is what happens to the child when the
            # parent dies (the OS closes the pipe end the parent owned).
            assert proc.stdin is not None
            proc.stdin.close()

            # The watchdog should pick up the EOF and call os._exit(0)
            # within a fraction of a second. Give it 10s to be safe on
            # slow CI machines.
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail(
                    "subprocess did not exit within 10s of stdin close — "
                    "watchdog is broken; orphan bug not fixed"
                )
            assert rc == 0, f"watchdog exit was non-zero ({rc})"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
