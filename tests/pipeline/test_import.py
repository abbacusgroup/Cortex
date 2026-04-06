"""Tests for CortexV1Importer and ObsidianImporter."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.importer import CortexV1Importer, ObsidianImporter


@pytest.fixture
def store(tmp_path):
    config = CortexConfig(data_dir=tmp_path / "data")
    s = Store(config)
    s.initialize(find_ontology())
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_v1_db(path: Path) -> None:
    """Create a minimal Cortex v1 SQLite database."""
    db = sqlite3.connect(str(path))
    db.execute(
        """CREATE TABLE documents (
            id TEXT PRIMARY KEY, title TEXT, content TEXT, type TEXT,
            project TEXT, tags TEXT, created_at TEXT
        )"""
    )
    db.execute(
        """INSERT INTO documents VALUES (
            'v1-1', 'Test Fix', 'Fixed a bug', 'fix',
            'myproj', 'bug', '2026-01-01'
        )"""
    )
    db.execute(
        """INSERT INTO documents VALUES (
            'v1-2', 'A Decision', 'We chose X', 'decision',
            'myproj', '', '2026-01-02'
        )"""
    )
    db.commit()
    db.close()


def _create_vault(path: Path) -> None:
    """Create a minimal Obsidian-style vault with markdown files."""
    path.mkdir(parents=True, exist_ok=True)

    (path / "fixes").mkdir()
    (path / "fixes" / "bug-fix.md").write_text(
        "---\ntype: fix\nproject: app\n---\nFixed a crash"
    )

    (path / "ideas").mkdir()
    (path / "ideas" / "new-feature.md").write_text(
        "# New Feature\nLet's build X"
    )


# ---------------------------------------------------------------------------
# CortexV1Importer
# ---------------------------------------------------------------------------


class TestCortexV1Import:
    """Tests for CortexV1Importer.run."""

    def test_import_valid_db(self, store, tmp_path):
        db_path = tmp_path / "v1.db"
        _create_v1_db(db_path)

        result = CortexV1Importer(store).run(db_path)

        assert result["status"] == "ok"
        assert result["imported"] == 2
        assert result["skipped"] == 0
        assert result["failed"] == 0
        assert result["total"] == 2

    def test_import_twice_deduplicates(self, store, tmp_path):
        db_path = tmp_path / "v1.db"
        _create_v1_db(db_path)

        CortexV1Importer(store).run(db_path)
        result = CortexV1Importer(store).run(db_path)

        assert result["status"] == "ok"
        assert result["imported"] == 0
        assert result["skipped"] == 2

    def test_nonexistent_file_returns_error(self, store, tmp_path):
        result = CortexV1Importer(store).run(tmp_path / "nope.db")

        assert result["status"] == "error"

    def test_type_mapping(self, store, tmp_path):
        """V1 types are mapped correctly to v2 ontology types."""
        db_path = tmp_path / "typed.db"
        db = sqlite3.connect(str(db_path))
        db.execute(
            """CREATE TABLE documents (
                id TEXT PRIMARY KEY, title TEXT, content TEXT, type TEXT,
                project TEXT, tags TEXT, created_at TEXT
            )"""
        )
        rows = [
            ("t1", "Capture", "body", "capture", "", "", "2026-01-01"),
            ("t2", "Note", "body", "note", "", "", "2026-01-02"),
            ("t3", "Guide", "body", "guide", "", "", "2026-01-03"),
            ("t4", "Workflow", "body", "workflow", "", "", "2026-01-04"),
        ]
        db.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        db.commit()
        db.close()

        result = CortexV1Importer(store).run(db_path)
        assert result["imported"] == 4

        # Verify mapped types via SQLite content store
        docs = store.content.list_documents(limit=100)
        types = {d["title"]: d["type"] for d in docs}

        assert types["Capture"] == "idea"  # capture -> idea
        assert types["Note"] == "idea"  # note -> idea
        assert types["Guide"] == "research"  # guide -> research
        assert types["Workflow"] == "session"  # workflow -> session


# ---------------------------------------------------------------------------
# ObsidianImporter
# ---------------------------------------------------------------------------


class TestObsidianImport:
    """Tests for ObsidianImporter.run."""

    def test_import_vault(self, store, tmp_path):
        vault = tmp_path / "vault"
        _create_vault(vault)

        result = ObsidianImporter(store).run(vault)

        assert result["status"] == "ok"
        assert result["imported"] == 2
        assert result["failed"] == 0

    def test_import_twice_deduplicates(self, store, tmp_path):
        vault = tmp_path / "vault"
        _create_vault(vault)

        ObsidianImporter(store).run(vault)
        result = ObsidianImporter(store).run(vault)

        assert result["status"] == "ok"
        assert result["imported"] == 0
        assert result["skipped"] == 2

    def test_empty_vault_zero_imported(self, store, tmp_path):
        vault = tmp_path / "empty_vault"
        vault.mkdir()

        result = ObsidianImporter(store).run(vault)

        assert result["status"] == "ok"
        assert result["imported"] == 0
        assert result["total"] == 0

    def test_non_directory_returns_error(self, store, tmp_path):
        fake = tmp_path / "not-a-dir.txt"
        fake.write_text("hi")

        result = ObsidianImporter(store).run(fake)

        assert result["status"] == "error"

    def test_nonexistent_path_returns_error(self, store, tmp_path):
        result = ObsidianImporter(store).run(tmp_path / "nope")

        assert result["status"] == "error"

    def test_frontmatter_parsed(self, store, tmp_path):
        """Frontmatter type, project, and tags are extracted."""
        vault = tmp_path / "fm_vault"
        vault.mkdir()
        (vault / "tagged.md").write_text(
            "---\ntype: lesson\nproject: cortex\ntags: [a, b]\n"
            "---\nLesson learned"
        )

        ObsidianImporter(store).run(vault)

        docs = store.content.list_documents(limit=10)
        assert len(docs) == 1
        doc = docs[0]
        assert doc["type"] == "lesson"
        assert doc["project"] == "cortex"
        assert "a" in doc["tags"]
        assert "b" in doc["tags"]

    def test_type_inferred_from_directory(self, store, tmp_path):
        """When frontmatter has no type, directory name is used."""
        vault = tmp_path / "dir_vault"
        (vault / "decisions").mkdir(parents=True)
        (vault / "decisions" / "pick-db.md").write_text(
            "We chose SQLite."
        )

        ObsidianImporter(store).run(vault)

        docs = store.content.list_documents(limit=10)
        assert len(docs) == 1
        assert docs[0]["type"] == "decision"

    def test_project_inferred_from_top_level_dir(self, store, tmp_path):
        """Top-level directory name becomes the project when not set."""
        vault = tmp_path / "proj_vault"
        (vault / "myproject" / "ideas").mkdir(parents=True)
        (vault / "myproject" / "ideas" / "cool.md").write_text(
            "A cool idea"
        )

        ObsidianImporter(store).run(vault)

        docs = store.content.list_documents(limit=10)
        assert len(docs) == 1
        assert docs[0]["project"] == "myproject"
