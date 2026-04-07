"""CLI tests for Phase 7 commands: setup, import-v1, import-vault."""

from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

import cortex.cli.main as cli_mod
from cortex.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_store(tmp_path, monkeypatch):
    """Reset module-level store and point CORTEX_DATA_DIR to tmp_path.

    Phase 3: also forces direct-store mode (instead of routing through MCP).
    """
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli_mod, "_use_mcp", lambda: False)
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class TestSetup:
    def test_setup_auto_exits_ok(self):
        result = runner.invoke(app, ["setup", "--auto"])
        assert result.exit_code == 0
        assert "ready" in result.output.lower()

    def test_setup_creates_data_dir(self, tmp_path):
        data_dir = tmp_path / "fresh"
        result = runner.invoke(
            app, ["setup", "--auto"], env={"CORTEX_DATA_DIR": str(data_dir)}
        )
        assert result.exit_code == 0
        assert data_dir.exists()


# ---------------------------------------------------------------------------
# Import V1
# ---------------------------------------------------------------------------


class TestImportV1:
    @staticmethod
    def _make_v1_db(path):
        """Create a minimal v1 SQLite database."""
        db = sqlite3.connect(str(path))
        db.execute(
            "CREATE TABLE documents "
            "(id TEXT, title TEXT, content TEXT, type TEXT, "
            "project TEXT, tags TEXT, created_at TEXT)"
        )
        db.execute(
            "INSERT INTO documents VALUES "
            "('1','Test Fix','content','fix','proj','',datetime('now'))"
        )
        db.commit()
        db.close()

    def test_import_v1_succeeds(self, tmp_path):
        db_path = tmp_path / "v1.db"
        self._make_v1_db(db_path)

        # Initialize store first
        runner.invoke(app, ["init"])

        result = runner.invoke(app, ["import-v1", str(db_path)])
        assert result.exit_code == 0
        assert "Imported: 1" in result.output

    def test_import_v1_nonexistent(self):
        # Initialize store first
        runner.invoke(app, ["init"])

        result = runner.invoke(app, ["import-v1", "/nonexistent"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Import Vault
# ---------------------------------------------------------------------------


class TestImportVault:
    def test_import_vault_succeeds(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "test.md").write_text("# Test\nSome content")

        # Initialize store first
        runner.invoke(app, ["init"])

        result = runner.invoke(app, ["import-vault", str(vault)])
        assert result.exit_code == 0
        assert "Imported: 1" in result.output

    def test_import_vault_nonexistent(self):
        # Initialize store first
        runner.invoke(app, ["init"])

        result = runner.invoke(app, ["import-vault", "/nonexistent"])
        assert result.exit_code == 1
