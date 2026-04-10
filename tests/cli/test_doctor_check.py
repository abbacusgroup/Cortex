"""Tests for ``cortex doctor check`` and ``cortex doctor repair`` (S4 + S5).

Covers:
- FTS5 integrity check and rebuild on ContentStore
- ReasonStage.check_fixpoint() read-only validation
- CLI doctor check/repair command behavior
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pyoxigraph as ox
import pytest
from typer.testing import CliRunner

import cortex.cli.main as cli_mod
from cortex.cli.main import app
from cortex.db.content_store import ContentStore
from cortex.db.graph_store import GraphStore
from cortex.ontology.namespaces import cortex_iri
from cortex.pipeline.reason import ReasonStage

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch):
    """Point the CLI at an isolated data dir and reset cached singletons."""
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False


@pytest.fixture()
def content_store(tmp_path: Path) -> ContentStore:
    """A real ContentStore with a few documents."""
    store = ContentStore(path=tmp_path / "cortex.db")
    for i in range(5):
        store.insert(
            doc_id=f"doc-{i}",
            title=f"Title {i}",
            content=f"Content for document {i}",
            tags=f"tag{i}",
        )
    yield store
    store.close()


@pytest.fixture()
def graph_store(tmp_path: Path) -> GraphStore:
    """A real GraphStore with ontology loaded."""
    from cortex.ontology.resolver import find_ontology

    store = GraphStore(path=tmp_path / "graph.db")
    store.load_ontology(find_ontology())
    yield store
    store.close()


def _create_full_stores(tmp_path: Path) -> None:
    """Create both stores for CLI tests (stores are opened/closed by commands)."""
    from cortex.ontology.resolver import find_ontology

    cs = ContentStore(path=tmp_path / "cortex.db")
    for i in range(3):
        cs.insert(doc_id=f"doc-{i}", title=f"Title {i}", content=f"Content {i}")
    cs.close()

    gs = GraphStore(path=tmp_path / "graph.db")
    gs.load_ontology(find_ontology())
    gs.close()


def _break_fts(db_path: Path) -> None:
    """Break FTS5 consistency by deleting a document without trigger."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TRIGGER IF EXISTS documents_ad")
    conn.execute("DELETE FROM documents WHERE id = 'doc-0'")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# FTS5 integrity check
# ---------------------------------------------------------------------------


class TestFtsIntegrityCheck:
    def test_consistent_store_reports_ok(self, content_store: ContentStore):
        result = content_store.fts_integrity_check()
        assert result["ok"] is True
        assert result["documents_count"] == 5
        assert result["error"] is None

    def test_detects_desync_after_triggerless_delete(
        self, content_store: ContentStore
    ):
        # Drop the AFTER DELETE trigger so FTS won't be notified
        content_store._db.execute("DROP TRIGGER IF EXISTS documents_ad")
        content_store._db.execute("DELETE FROM documents WHERE id = 'doc-0'")
        content_store._db.commit()

        result = content_store.fts_integrity_check()
        assert result["ok"] is False
        assert result["error"] is not None

    def test_detects_desync_after_triggerless_insert(
        self, content_store: ContentStore
    ):
        # Drop the AFTER INSERT trigger so FTS won't index the new row
        content_store._db.execute("DROP TRIGGER IF EXISTS documents_ai")
        content_store._db.execute(
            "INSERT INTO documents (id, title, content, type, created_at, updated_at) "
            "VALUES ('extra', 'Extra', 'No FTS', 'idea', '2026-04-10', '2026-04-10')"
        )
        content_store._db.commit()

        result = content_store.fts_integrity_check()
        assert result["ok"] is False
        assert result["error"] is not None

    def test_empty_store_is_ok(self, tmp_path: Path):
        store = ContentStore(path=tmp_path / "empty.db")
        result = store.fts_integrity_check()
        assert result["ok"] is True
        assert result["documents_count"] == 0
        store.close()


# ---------------------------------------------------------------------------
# FTS5 rebuild
# ---------------------------------------------------------------------------


class TestFtsRebuild:
    def test_rebuild_fixes_desync(self, content_store: ContentStore):
        # Break FTS by removing a document without trigger
        content_store._db.execute("DROP TRIGGER IF EXISTS documents_ad")
        content_store._db.execute("DELETE FROM documents WHERE id = 'doc-0'")
        content_store._db.commit()

        check = content_store.fts_integrity_check()
        assert not check["ok"]

        # Rebuild should fix it
        result = content_store.fts_rebuild()
        assert result["rebuilt"] is True
        assert result["documents_count"] == 4

        check = content_store.fts_integrity_check()
        assert check["ok"]

    def test_rebuild_makes_missing_docs_searchable(
        self, content_store: ContentStore
    ):
        # Insert bypassing trigger
        content_store._db.execute("DROP TRIGGER IF EXISTS documents_ai")
        content_store._db.execute(
            "INSERT INTO documents (id, title, content, type, created_at, updated_at) "
            "VALUES ('extra', 'Extra Doc', 'uniquefindme', 'idea', "
            "'2026-04-10', '2026-04-10')"
        )
        content_store._db.commit()

        # Not searchable yet (FTS doesn't know about it)
        results = content_store.search("uniquefindme")
        assert len(results) == 0

        content_store.fts_rebuild()

        # Now searchable
        results = content_store.search("uniquefindme")
        assert len(results) == 1

    def test_rebuild_is_idempotent(self, content_store: ContentStore):
        r1 = content_store.fts_rebuild()
        r2 = content_store.fts_rebuild()
        assert r1["documents_count"] == r2["documents_count"]

        check = content_store.fts_integrity_check()
        assert check["ok"]


# ---------------------------------------------------------------------------
# Reasoner check_fixpoint
# ---------------------------------------------------------------------------


class TestReasonerCheckFixpoint:
    def test_fixpoint_returns_zero(self, graph_store: GraphStore):
        reasoner = ReasonStage(graph_store)
        # Run once to reach fixpoint
        reasoner.run()
        # Check should show zero pending
        result = reasoner.check_fixpoint()
        assert result["ok"] is True
        assert result["total_pending"] == 0

    def test_detects_missing_symmetric(self, graph_store: GraphStore):
        # Add A contradicts B without the reverse
        a = cortex_iri("obj-a")
        b = cortex_iri("obj-b")
        contradicts = ox.NamedNode(cortex_iri("contradicts").value)
        graph_store._store.add(ox.Quad(a, contradicts, b))

        reasoner = ReasonStage(graph_store)
        result = reasoner.check_fixpoint()
        assert result["ok"] is False
        assert result["total_pending"] > 0
        assert result["rule_counts"]["symmetric_contradicts"] >= 1

    def test_check_does_not_write(self, graph_store: GraphStore):
        # Add A contradicts B without reverse
        a = cortex_iri("obj-a")
        b = cortex_iri("obj-b")
        contradicts = ox.NamedNode(cortex_iri("contradicts").value)
        graph_store._store.add(ox.Quad(a, contradicts, b))

        # Count triples before
        before = len(list(graph_store._store.quads_for_pattern(None, None, None)))

        reasoner = ReasonStage(graph_store)
        reasoner.check_fixpoint()

        # Count after — should be unchanged
        after = len(list(graph_store._store.quads_for_pattern(None, None, None)))
        assert before == after


# ---------------------------------------------------------------------------
# doctor check CLI
# ---------------------------------------------------------------------------


class TestDoctorCheck:
    def test_check_reports_all_ok(self, tmp_path: Path):
        _create_full_stores(tmp_path)
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0, result.output
        assert "OK" in result.output
        assert "All checks passed" in result.output

    def test_check_reports_fts_warning(self, tmp_path: Path):
        _create_full_stores(tmp_path)
        _break_fts(tmp_path / "cortex.db")

        result = runner.invoke(app, ["doctor", "check"])
        assert "WARN" in result.output
        assert "inconsistent" in result.output

    def test_check_fails_without_stores(self, tmp_path: Path):
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


# ---------------------------------------------------------------------------
# doctor repair CLI
# ---------------------------------------------------------------------------


class TestDoctorRepair:
    def test_repair_rebuilds_fts(self, tmp_path: Path):
        _create_full_stores(tmp_path)
        _break_fts(tmp_path / "cortex.db")

        result = runner.invoke(app, ["doctor", "repair"])
        assert result.exit_code == 0, result.output
        assert "rebuilt" in result.output
        assert "Repair complete" in result.output

    def test_repair_reaches_fixpoint(self, tmp_path: Path):
        _create_full_stores(tmp_path)
        result = runner.invoke(app, ["doctor", "repair"])
        assert result.exit_code == 0, result.output
        assert "Reasoner" in result.output
        assert "fixpoint" in result.output or "triples inferred" in result.output

    def test_repair_refuses_running_server(self, tmp_path: Path):
        import json
        import os

        _create_full_stores(tmp_path)
        marker = tmp_path / "graph.db.lock"
        marker.write_text(json.dumps({
            "pid": os.getpid(),
            "cmdline": "cortex serve --transport mcp-http",
            "acquired_at": "2026-04-10T00:00:00+00:00",
        }))

        result = runner.invoke(app, ["doctor", "repair"])
        assert result.exit_code == 1
        assert "running" in (result.output + (result.stderr or "")).lower()

    def test_repair_checkpoints_wal(self, tmp_path: Path):
        _create_full_stores(tmp_path)
        result = runner.invoke(app, ["doctor", "repair"])
        assert result.exit_code == 0
        assert "checkpointed" in result.output
