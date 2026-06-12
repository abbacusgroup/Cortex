"""CLI integration tests for Cortex.

Tests all CLI commands via Typer's CliRunner with isolated tmp_path data dirs.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

import cortex.cli.main as cli_mod
from cortex.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Reset module-level store and point CORTEX_DATA_DIR to tmp_path.

    Phase 3: also forces direct-store mode (instead of routing through the
    MCP HTTP server). These tests rely on per-test ``tmp_path`` isolation,
    which only works against the direct store — MCP routing would hit the
    user's real production data.
    """
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CORTEX_TEST_MODE", "1")
    monkeypatch.setattr(cli_mod, "_use_mcp", lambda: False)
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_fix(title: str = "Test Fix", content: str = "fix content"):
    """Capture a fix and return the result."""
    return runner.invoke(app, ["capture", title, "--type", "fix", "--content", content])


def _extract_id(output: str) -> str:
    """Extract the object ID from capture output like 'Captured fix: <id>'."""
    match = re.search(r"Captured \w+: (\S+)", output)
    assert match, f"Could not extract ID from: {output!r}"
    return match.group(1)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    """Test that ``cortex init`` (deprecated) delegates to the setup wizard."""

    def test_init_shows_deprecation_notice(self):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "deprecated" in result.output

    def test_init_runs_wizard(self):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Cortex Setup" in result.output
        assert "Cortex is ready" in result.output

    def test_init_shows_embeddings_status(self):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        # Wizard mentions embeddings — either model name or "NOT installed"
        assert "Embeddings" in result.output

    def test_init_warmup_reports_not_installed(self, monkeypatch):
        """When sentence-transformers is absent, init reports it gracefully."""
        import builtins

        real_import = builtins.__import__

        def _block_st(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_st)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "not installed" in result.output.lower()


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class TestCapture:
    def test_capture_fix(self):
        result = _capture_fix()
        assert result.exit_code == 0
        assert "Captured fix:" in result.output
        # Output should contain a UUID-like ID
        obj_id = _extract_id(result.output)
        assert len(obj_id) > 8

    def test_capture_invalid_type(self):
        result = runner.invoke(
            app, ["capture", "Bad", "--type", "nonexistent_type", "--content", "x"]
        )
        assert result.exit_code == 1

    def test_capture_no_content_no_stdin(self):
        result = runner.invoke(app, ["capture", "Empty", "--type", "idea"], input="")
        assert result.exit_code == 1
        assert "no content" in result.output.lower()

    def test_capture_with_stdin(self):
        result = runner.invoke(
            app, ["capture", "From Stdin", "--type", "idea"], input="piped content"
        )
        assert result.exit_code == 0
        assert "Captured idea:" in result.output


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_captured(self):
        cap = _capture_fix(title="Unique Widget Fix", content="fixed the widget")
        assert cap.exit_code == 0

        result = runner.invoke(app, ["search", "Widget"])
        assert result.exit_code == 0
        assert "Widget" in result.output

    def test_search_no_results(self):
        result = runner.invoke(app, ["search", "zzz_nonexistent_zzz"])
        assert result.exit_code == 0
        assert "no results" in result.output.lower()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


class TestRead:
    def test_read_captured_object(self):
        cap = _capture_fix(title="Readable Fix", content="detailed content")
        assert cap.exit_code == 0
        obj_id = _extract_id(cap.output)

        result = runner.invoke(app, ["read", obj_id])
        assert result.exit_code == 0
        assert "Readable Fix" in result.output
        assert "detailed content" in result.output

    def test_read_nonexistent(self):
        # Ensure store is initialized first
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["read", "nonexistent-id-00000"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestList:
    def test_list_shows_captured(self):
        r1 = _capture_fix(title="Fix Alpha", content="alpha content")
        assert r1.exit_code == 0
        r2 = runner.invoke(
            app, ["capture", "Idea Beta", "--type", "idea", "--content", "beta content"]
        )
        assert r2.exit_code == 0

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "Beta" in result.output

    def test_list_filter_by_type(self):
        r1 = _capture_fix(title="Fix Only", content="fix stuff")
        assert r1.exit_code == 0
        r2 = runner.invoke(
            app, ["capture", "Idea Only", "--type", "idea", "--content", "idea stuff"]
        )
        assert r2.exit_code == 0

        result = runner.invoke(app, ["list", "--type", "fix"])
        assert result.exit_code == 0
        assert "Fix Only" in result.output
        assert "Idea Only" not in result.output


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_shows_counts(self):
        _capture_fix(title="Status Fix", content="for status test")

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Documents:" in result.output or "documents" in result.output.lower()


# ---------------------------------------------------------------------------
# Short-id prefixes (read / graph / pipeline accept the 8-char ids shown
# by list/search)
# ---------------------------------------------------------------------------


def _seed_doc(tmp_path, doc_id: str, title: str = "Seeded") -> None:
    """Insert a document with a controlled id directly into the content store."""
    from cortex.db.content_store import ContentStore

    cs = ContentStore(path=tmp_path / "cortex.db")
    cs.insert(doc_id=doc_id, title=title, content="seeded body")
    cs.close()


class TestShortIdPrefixes:
    def test_read_accepts_short_id_prefix(self):
        cap = _capture_fix(title="Short Id Fix")
        full_id = _extract_id(cap.output)

        result = runner.invoke(app, ["read", full_id[:8]])
        assert result.exit_code == 0
        assert full_id in result.output
        assert "Short Id Fix" in result.output

    def test_read_full_uuid_unchanged(self):
        cap = _capture_fix(title="Full Id Fix")
        full_id = _extract_id(cap.output)

        result = runner.invoke(app, ["read", full_id])
        assert result.exit_code == 0
        assert "Full Id Fix" in result.output

    def test_read_unknown_prefix_still_not_found(self):
        _capture_fix()
        result = runner.invoke(app, ["read", "zzzzzzzz"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_read_ambiguous_prefix_lists_candidates(self, tmp_path):
        _seed_doc(tmp_path, "aaaa1111-0000-0000-0000-000000000000", title="One")
        _seed_doc(tmp_path, "aaaa2222-0000-0000-0000-000000000000", title="Two")

        result = runner.invoke(app, ["read", "aaaa"])
        assert result.exit_code == 1
        assert "Ambiguous" in result.output
        assert "aaaa1111-0000-0000-0000-000000000000" in result.output
        assert "aaaa2222-0000-0000-0000-000000000000" in result.output

    def test_graph_accepts_short_id_prefix(self):
        cap = _capture_fix(title="Graph Short Id Fix")
        full_id = _extract_id(cap.output)

        result = runner.invoke(app, ["graph", full_id[:8]])
        assert result.exit_code == 0

    def test_pipeline_accepts_short_id_prefix(self):
        cap = _capture_fix(title="Pipeline Short Id Fix")
        full_id = _extract_id(cap.output)

        result = runner.invoke(app, ["pipeline", full_id[:8]])
        assert result.exit_code == 0
        # Output names the resolved full id, not the prefix
        assert f"for {full_id[:12]}" in result.output


# ---------------------------------------------------------------------------
# Validated inputs (graph existence, entities --type, install --service)
# ---------------------------------------------------------------------------


class TestGraphExistenceCheck:
    def test_graph_bogus_id_exits_1(self):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["graph", "totally-bogus-id-12345"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_graph_existing_object_without_relationships_exits_0(self):
        cap = _capture_fix(title="Lonely Fix")
        full_id = _extract_id(cap.output)

        result = runner.invoke(app, ["graph", full_id])
        assert result.exit_code == 0
        assert "No relationships found." in result.output


class TestEntitiesTypeValidation:
    def test_unknown_type_exits_1_with_valid_list(self):
        result = runner.invoke(app, ["entities", "--type", "technologies"])
        assert result.exit_code == 1
        assert "Invalid entity type 'technologies'" in result.output
        for valid in ("concept", "pattern", "project", "technology"):
            assert valid in result.output

    def test_valid_type_is_case_insensitive(self):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["entities", "--type", "TECHNOLOGY"])
        assert result.exit_code == 0


class TestServiceValidation:
    def test_install_invalid_service_exits_1(self, monkeypatch):
        import cortex.cli.install as install_mod

        called: list[str] = []
        monkeypatch.setattr(
            install_mod, "do_install", lambda config, service: called.append(service)
        )
        result = runner.invoke(app, ["install", "--service", "bogus"])
        assert result.exit_code == 1
        assert "Invalid service 'bogus'" in result.output
        assert "all" in result.output and "dashboard" in result.output and "mcp" in result.output
        assert called == []

    def test_uninstall_invalid_service_exits_1(self, monkeypatch):
        import cortex.cli.install as install_mod

        called: list[str] = []
        monkeypatch.setattr(
            install_mod, "do_uninstall", lambda config, service: called.append(service)
        )
        result = runner.invoke(app, ["uninstall", "--service", "dashbord"])
        assert result.exit_code == 1
        assert "Invalid service 'dashbord'" in result.output
        assert called == []

    def test_install_valid_service_passes_through(self, monkeypatch):
        import cortex.cli.install as install_mod

        called: list[str] = []
        monkeypatch.setattr(
            install_mod, "do_install", lambda config, service: called.append(service)
        )
        result = runner.invoke(app, ["install", "--service", "mcp"])
        assert result.exit_code == 0
        assert called == ["mcp"]


# ---------------------------------------------------------------------------
# Full Round-Trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_capture_search_read_list(self):
        # 1. Capture
        cap = runner.invoke(
            app,
            ["capture", "Round Trip Item", "--type", "lesson", "--content", "learned this"],
        )
        assert cap.exit_code == 0
        obj_id = _extract_id(cap.output)

        # 2. Search
        search = runner.invoke(app, ["search", "Round Trip"])
        assert search.exit_code == 0
        assert "Round Trip" in search.output

        # 3. Read
        read = runner.invoke(app, ["read", obj_id])
        assert read.exit_code == 0
        assert "Round Trip Item" in read.output
        assert "learned this" in read.output

        # 4. List
        lst = runner.invoke(app, ["list"])
        assert lst.exit_code == 0
        assert "Round Trip" in lst.output
