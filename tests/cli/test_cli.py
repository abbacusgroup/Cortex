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
    """Reset module-level store and point CORTEX_DATA_DIR to tmp_path."""
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None


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
    def test_init_succeeds(self):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "initialized" in result.output.lower()

    def test_init_idempotent(self):
        first = runner.invoke(app, ["init"])
        assert first.exit_code == 0
        # Reset store so second init creates a fresh one
        cli_mod._store = None
        second = runner.invoke(app, ["init"])
        assert second.exit_code == 0
        assert "initialized" in second.output.lower()


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
