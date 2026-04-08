"""Per-command tests for Phase 3 MCP routing.

For each of the 12 MCP-routed CLI commands, verify:
- Default mode (no `--direct`) calls the corresponding MCP client method
- The command's stdout contains the expected fields from the mocked response
- A clean error is shown when the MCP client raises a connection error

The byte-identical contract between MCP and `--direct` modes is verified
separately in tests/cli/test_cli.py (which uses --direct mode for all
existing tests via the autouse fixture).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

import cortex.cli.main as cli_mod
from cortex.cli.main import app
from cortex.transport.mcp.client import MCPConnectionError

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    """Reset CLI state and point CORTEX_DATA_DIR at tmp_path."""
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._direct_mode = False
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._direct_mode = False
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False


def _install_fake_client(monkeypatch, **method_returns):
    """Install a fake MCP client that returns canned responses for the named methods.

    Each kwarg becomes an async method that returns its value. Also stubs
    ``list_tools`` so the lazy probe passes. Bundle 9 / D.2: the lazy
    probe now goes through ``_get_probe_client``, so we patch both helpers
    to point at the same fake.
    """
    fake = MagicMock()

    # Probe satisfies _REQUIRED_MCP_ROUTING_TOOLS
    fake.list_tools = AsyncMock(return_value=list(cli_mod._REQUIRED_MCP_ROUTING_TOOLS))

    for name, value in method_returns.items():
        setattr(fake, name, AsyncMock(return_value=value))

    monkeypatch.setattr(cli_mod, "_get_mcp_client", lambda: fake)
    monkeypatch.setattr(cli_mod, "_get_probe_client", lambda: fake)
    return fake


# ─── Read-only commands ────────────────────────────────────────────────────


class TestSearchCommand:
    def test_routes_to_cortex_search(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            search=[
                {"id": "abc123def456", "type": "fix", "title": "T1", "project": ""},
            ],
        )
        result = runner.invoke(app, ["search", "needle"])
        assert result.exit_code == 0
        fake.search.assert_called_once()
        kwargs = fake.search.call_args.kwargs
        assert kwargs["query"] == "needle"
        assert "T1" in result.output

    def test_unreachable_mcp_exits_clean(self, monkeypatch):
        fake = MagicMock()
        fake.list_tools = AsyncMock(side_effect=MCPConnectionError("nope"))
        # Bundle 9 / D.2: the lazy probe goes through ``_get_probe_client``
        # now (a separate short-timeout client), so we patch that as well
        # so the unreachable error fires from the probe path.
        monkeypatch.setattr(cli_mod, "_get_mcp_client", lambda: fake)
        monkeypatch.setattr(cli_mod, "_get_probe_client", lambda: fake)
        result = runner.invoke(app, ["search", "anything"])
        assert result.exit_code == 1
        assert "Cannot reach" in result.output
        assert "Traceback" not in result.output


class TestListCommand:
    def test_routes_to_cortex_list(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            list_objects=[
                {"id": "id1aaaaa", "type": "idea", "title": "X", "project": ""},
                {"id": "id2bbbbb", "type": "fix", "title": "Y", "project": "p"},
            ],
        )
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        fake.list_objects.assert_called_once()
        assert "X" in result.output
        assert "Y" in result.output


class TestStatusCommand:
    def test_routes_to_cortex_status(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            status={
                "initialized": True,
                "sqlite_total": 161,
                "graph_triples": 10473,
                "entities": 636,
                "counts_by_type": {"fix": 40, "decision": 28},
            },
        )
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        fake.status.assert_called_once()
        assert "161" in result.output
        assert "10473" in result.output
        assert "636" in result.output


class TestReadCommand:
    def test_routes_to_cortex_read(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            read={
                "id": "abc",
                "type": "lesson",
                "title": "Test Lesson",
                "project": "p",
                "tags": "a,b",
                "tier": "recall",
                "created_at": "2026-04-07",
                "content": "lesson body",
                "relationships": [],
            },
        )
        result = runner.invoke(app, ["read", "abc"])
        assert result.exit_code == 0
        fake.read.assert_called_once_with("abc")
        assert "Test Lesson" in result.output
        assert "lesson body" in result.output

    def test_not_found_exits_1(self, monkeypatch):
        _install_fake_client(monkeypatch, read="Not found: missing-id")
        result = runner.invoke(app, ["read", "missing-id"])
        assert result.exit_code == 1
        assert "Not found" in result.output


class TestContextCommand:
    def test_routes_to_cortex_context(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            context=[
                {
                    "type": "fix",
                    "title": "Brief1",
                    "summary": "Short summary text",
                    "score": 0.8,
                },
            ],
        )
        result = runner.invoke(app, ["context", "topic"])
        assert result.exit_code == 0
        fake.context.assert_called_once()
        assert "Brief1" in result.output


class TestDossierCommand:
    def test_routes_to_cortex_dossier(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            dossier={
                "topic": "SQLite",
                "entity": {"name": "sqlite", "type": "technology"},
                "object_count": 9,
                "objects": [
                    {"type": "fix", "title": "Lock fix", "summary": "fix summary"},
                ],
                "contradictions": [],
                "related_entities": [],
            },
        )
        result = runner.invoke(app, ["dossier", "SQLite"])
        assert result.exit_code == 0
        fake.dossier.assert_called_once_with(topic="SQLite")
        assert "SQLite" in result.output
        assert "Lock fix" in result.output


class TestGraphCommand:
    def test_routes_to_cortex_graph(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            graph={
                "causal_chain": [
                    {"type": "fix", "title": "A"},
                    {"type": "fix", "title": "B"},
                ],
                "evolution": [],
                "relationships": [
                    {"direction": "outgoing", "rel_type": "supports", "other_id": "abc12345xx"},
                ],
            },
        )
        result = runner.invoke(app, ["graph", "abc"])
        assert result.exit_code == 0
        fake.graph.assert_called_once_with(obj_id="abc")
        assert "Causal chain" in result.output
        assert "supports" in result.output


class TestEntitiesCommand:
    def test_routes_to_cortex_list_entities(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            list_entities=[
                {"id": "e1aaaaaa", "type": "technology", "name": "sqlite"},
                {"id": "e2bbbbbb", "type": "concept", "name": "lock"},
            ],
        )
        result = runner.invoke(app, ["entities"])
        assert result.exit_code == 0
        fake.list_entities.assert_called_once()
        assert "sqlite" in result.output
        assert "lock" in result.output


# ─── Write commands ────────────────────────────────────────────────────────


class TestCaptureCommand:
    def test_routes_to_cortex_capture(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch, capture={"id": "new-obj-id-12345", "status": "complete"}
        )
        result = runner.invoke(
            app,
            [
                "capture",
                "Test title",
                "--type",
                "idea",
                "--content",
                "test body",
            ],
        )
        assert result.exit_code == 0
        fake.capture.assert_called_once()
        kwargs = fake.capture.call_args.kwargs
        assert kwargs["title"] == "Test title"
        assert kwargs["content"] == "test body"
        assert kwargs["obj_type"] == "idea"
        assert "new-obj-id-12345" in result.output


# ─── Admin commands ────────────────────────────────────────────────────────


class TestSynthesizeCommand:
    def test_routes_to_cortex_synthesize(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            synthesize={
                "object_count": 5,
                "themes": [{"name": "auth", "count": 3}],
                "narrative": "five things happened",
            },
        )
        result = runner.invoke(app, ["synthesize", "--period", "7"])
        assert result.exit_code == 0
        fake.synthesize.assert_called_once()
        kwargs = fake.synthesize.call_args.kwargs
        assert kwargs["period_days"] == 7
        assert "five things" in result.output


class TestPipelineCommand:
    def test_routes_to_cortex_pipeline(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            pipeline={
                "id": "obj-id-1234567890",
                "status": "complete",
                "pipeline_stages": {
                    "normalize": {"status": "normalized"},
                    "link": {"status": "linked"},
                },
            },
        )
        result = runner.invoke(app, ["pipeline", "obj-id-1234567890"])
        assert result.exit_code == 0
        fake.pipeline.assert_called_once_with(obj_id="obj-id-1234567890")
        assert "complete" in result.output
        assert "normalize" in result.output

    def test_batch_requires_direct(self, monkeypatch):
        # batch mode is direct-only — without --direct it should error
        _install_fake_client(monkeypatch)
        result = runner.invoke(app, ["pipeline", "--batch"])
        assert result.exit_code == 1
        assert "--direct" in result.output


class TestReasonCommand:
    def test_routes_to_cortex_reason(self, monkeypatch):
        fake = _install_fake_client(
            monkeypatch,
            reason={
                "contradictions": [{"severity": "medium", "message": "X conflicts Y"}],
                "patterns": [],
                "gaps": [],
                "staleness": [],
            },
        )
        result = runner.invoke(app, ["reason"])
        assert result.exit_code == 0
        fake.reason.assert_called_once()
        assert "contradictions" in result.output
        assert "X conflicts Y" in result.output


# ─── Cross-cutting: --direct must reset between invocations ───────────────


class TestDirectFlagInteraction:
    def test_direct_mode_skips_mcp_client(self, monkeypatch):
        """When --direct is set, _get_mcp_client should NEVER be called."""
        fake = MagicMock(side_effect=AssertionError("MCP client should not be called"))
        monkeypatch.setattr(cli_mod, "_get_mcp_client", fake)
        # Mock _get_store to avoid actually opening one
        with monkeypatch.context() as m:
            mock_store = MagicMock()
            mock_store.list_objects.return_value = []
            mock_get_store = MagicMock(return_value=mock_store)
            m.setattr(cli_mod, "_get_store", mock_get_store)
            result = runner.invoke(app, ["--direct", "list"])
        assert result.exit_code == 0
        fake.assert_not_called()


# ─── register command (Phase 3 bonus) ─────────────────────────────────────


class TestRegisterCommand:
    """register defaults to writing the new HTTP transport entry; --legacy-stdio
    keeps the pre-Phase-2 stdio behavior.
    """

    def test_register_default_writes_http_transport(self, tmp_path, monkeypatch):
        import json

        # Sandbox ~/.claude/settings.json into the test's tmp_path
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        result = runner.invoke(app, ["register"])
        assert result.exit_code == 0
        settings_path = fake_home / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        cortex_entry = settings["mcpServers"]["cortex"]
        assert cortex_entry["type"] == "http"
        assert "url" in cortex_entry
        assert "127.0.0.1:1314/mcp" in cortex_entry["url"]
        # No stdio command/args fields
        assert "command" not in cortex_entry
        assert "args" not in cortex_entry
        assert "Restart Claude Code" in result.output

    def test_register_legacy_stdio_writes_old_format(self, tmp_path, monkeypatch):
        import json

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        result = runner.invoke(app, ["register", "--legacy-stdio"])
        assert result.exit_code == 0
        settings_path = fake_home / ".claude" / "settings.json"
        settings = json.loads(settings_path.read_text())
        cortex_entry = settings["mcpServers"]["cortex"]
        # Legacy format: command + args, no type
        assert "command" in cortex_entry
        assert "args" in cortex_entry
        # The exact args depend on whether `cortex` is on PATH:
        #   - on PATH:    args = ["serve", "--transport", "stdio"]
        #   - not on PATH: args = ["-m", "cortex.transport.mcp"]
        # Either way, "stdio" should NOT be in the type field (no type field at all).
        assert "type" not in cortex_entry
        assert "stdio" in result.output.lower()

    def test_register_preserves_other_mcp_servers(self, tmp_path, monkeypatch):
        import json

        fake_home = tmp_path / "home"
        (fake_home / ".claude").mkdir(parents=True)
        existing = {
            "mcpServers": {
                "other-server": {"command": "other-bin", "args": []},
            }
        }
        (fake_home / ".claude" / "settings.json").write_text(json.dumps(existing))
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        runner.invoke(app, ["register"])
        settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
        # Both entries present, only cortex was added/updated
        assert "other-server" in settings["mcpServers"]
        assert "cortex" in settings["mcpServers"]
        assert settings["mcpServers"]["other-server"]["command"] == "other-bin"

    def test_register_overwrites_existing_cortex_entry(self, tmp_path, monkeypatch):
        import json

        fake_home = tmp_path / "home"
        (fake_home / ".claude").mkdir(parents=True)
        existing = {
            "mcpServers": {
                "cortex": {
                    "command": "/old/path/cortex",
                    "args": ["serve", "--transport", "stdio"],
                }
            }
        }
        (fake_home / ".claude" / "settings.json").write_text(json.dumps(existing))
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        runner.invoke(app, ["register"])
        settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
        cortex_entry = settings["mcpServers"]["cortex"]
        # Now HTTP, not stdio
        assert cortex_entry["type"] == "http"
        assert "command" not in cortex_entry
