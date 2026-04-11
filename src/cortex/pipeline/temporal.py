"""Temporal versioning — version history and state-at-time queries.

On update, the old version is preserved with a valid_to timestamp.
Supports querying the state of an object at any point in time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from cortex.core.logging import get_logger
from cortex.db.content_store import ContentStore

logger = get_logger("pipeline.temporal")

VERSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    version_num INTEGER NOT NULL,
    data TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_versions_doc_id
    ON document_versions(doc_id);
CREATE INDEX IF NOT EXISTS idx_versions_valid
    ON document_versions(doc_id, valid_from, valid_to);
"""


class TemporalVersioning:
    """Version history for knowledge objects."""

    def __init__(self, content_store: ContentStore):
        self.db = content_store._db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.executescript(VERSIONS_TABLE_SQL)
        self.db.commit()

    def snapshot_before_update(self, doc_id: str) -> int | None:
        """Save the current state as a version before an update.

        Returns:
            The version number, or None if document not found.
        """
        row = self.db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None

        data = dict(row)
        now = datetime.now(UTC).isoformat()

        # Get next version number
        last = self.db.execute(
            "SELECT MAX(version_num) as v FROM document_versions WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        version_num = (last["v"] or 0) + 1

        self.db.execute(
            """INSERT INTO document_versions
               (doc_id, version_num, data, valid_from, valid_to, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                version_num,
                json.dumps(data, default=str),
                data.get("updated_at", data.get("created_at", now)),
                now,
                now,
            ),
        )
        self.db.commit()
        return version_num

    def get_version(self, doc_id: str, version_num: int) -> dict[str, Any] | None:
        """Get a specific version of a document."""
        row = self.db.execute(
            "SELECT data FROM document_versions WHERE doc_id = ? AND version_num = ?",
            (doc_id, version_num),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

    def get_state_at(self, doc_id: str, at_time: str) -> dict[str, Any] | None:
        """Get the state of a document at a specific time.

        Args:
            doc_id: Document ID.
            at_time: ISO format datetime string.

        Returns:
            Document state at that time, or None if it didn't exist yet.
        """
        # Check versions (most recent valid_from <= at_time)
        row = self.db.execute(
            """SELECT data FROM document_versions
               WHERE doc_id = ? AND valid_from <= ? AND valid_to > ?
               ORDER BY version_num DESC LIMIT 1""",
            (doc_id, at_time, at_time),
        ).fetchone()
        if row:
            return json.loads(row["data"])

        # Maybe the current version is valid at that time
        current = self.db.execute(
            "SELECT * FROM documents WHERE id = ? AND created_at <= ?",
            (doc_id, at_time),
        ).fetchone()
        if current:
            return dict(current)

        return None

    def list_versions(self, doc_id: str) -> list[dict[str, Any]]:
        """List all versions of a document (oldest first).

        Returns:
            List of version metadata (version_num, valid_from, valid_to).
        """
        rows = self.db.execute(
            """SELECT version_num, valid_from, valid_to, created_at
               FROM document_versions
               WHERE doc_id = ?
               ORDER BY version_num ASC""",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def version_count(self, doc_id: str) -> int:
        """Get total version count for a document."""
        row = self.db.execute(
            "SELECT COUNT(*) as c FROM document_versions WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return row["c"]
