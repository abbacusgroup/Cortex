"""SQLite-backed content store for Cortex.

Handles content storage, FTS5 full-text search, config, and query logging.
Uses synchronous sqlite3 (async wrapper can be added later via aiosqlite).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortex.core.errors import NotFoundError, StoreError
from cortex.core.logging import get_logger

logger = get_logger("db.content")

SCHEMA_SQL = """
-- Main content table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    raw_markdown TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'capture',
    project TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    tier TEXT NOT NULL DEFAULT 'archive',
    pipeline_stage TEXT NOT NULL DEFAULT 'ingest',
    confidence REAL NOT NULL DEFAULT 1.0,
    captured_by TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- FTS5 virtual table (title weighted 10x, tags 5x, content 1x)
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    content,
    tags,
    content=documents,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content, tags)
    VALUES('delete', old.rowid, old.title, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content, tags)
    VALUES('delete', old.rowid, old.title, old.content, old.tags);
    INSERT INTO documents_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags);
END;

-- Config key-value store
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Query log
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    tool TEXT NOT NULL DEFAULT '',
    params TEXT NOT NULL DEFAULT '{}',
    result_ids TEXT NOT NULL DEFAULT '[]',
    result_count INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0,
    session_id TEXT NOT NULL DEFAULT ''
);

-- Embedding storage
CREATE TABLE IF NOT EXISTS embeddings (
    doc_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    dimensions INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_query_log_timestamp ON query_log(timestamp);
"""


class ContentStore:
    """SQLite store for document content, FTS5 search, config, and query logs."""

    def __init__(self, path: Path | None = None):
        """Initialize the store.

        Args:
            path: Path to SQLite database file. If None, uses in-memory.
        """
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
        else:
            self._db = sqlite3.connect(":memory:", check_same_thread=False)

        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._db.executescript(SCHEMA_SQL)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -------------------------------------------------------------------------
    # Document CRUD
    # -------------------------------------------------------------------------

    def insert(
        self,
        *,
        doc_id: str,
        title: str,
        content: str = "",
        raw_markdown: str = "",
        doc_type: str = "capture",
        project: str = "",
        tags: str = "",
        summary: str = "",
        tier: str = "archive",
        pipeline_stage: str = "ingest",
        confidence: float = 1.0,
        captured_by: str = "",
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> str:
        """Insert a document.

        Returns:
            The document ID.
        """
        now = datetime.now(UTC).isoformat()
        ts_created = created_at or now
        ts_updated = updated_at or now
        try:
            self._db.execute(
                """INSERT INTO documents
                   (id, title, content, raw_markdown, type, project, tags, summary,
                    tier, pipeline_stage, confidence, captured_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, title, content, raw_markdown, doc_type, project, tags, summary,
                 tier, pipeline_stage, confidence, captured_by, ts_created, ts_updated),
            )
            self._db.commit()
        except sqlite3.IntegrityError as e:
            raise StoreError(f"Document already exists: {doc_id}", cause=e) from e
        return doc_id

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID.

        Returns:
            Dict with all fields, or None if not found.
        """
        row = self._db.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def update(self, doc_id: str, **updates: Any) -> bool:
        """Update document fields.

        Returns:
            True if document existed and was updated.

        Raises:
            NotFoundError: If document doesn't exist.
        """
        existing = self.get(doc_id)
        if existing is None:
            raise NotFoundError(f"Document not found: {doc_id}", context={"id": doc_id})

        if not updates:
            return True

        updates["updated_at"] = datetime.now(UTC).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*updates.values(), doc_id]

        self._db.execute(
            f"UPDATE documents SET {set_clause} WHERE id = ?", values
        )
        self._db.commit()
        return True

    def delete(self, doc_id: str) -> bool:
        """Delete a document.

        Returns:
            True if a document was deleted.
        """
        cursor = self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._db.commit()
        return cursor.rowcount > 0

    def list_documents(
        self,
        *,
        doc_type: str | None = None,
        project: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List documents with optional filters."""
        conditions = []
        params: list[Any] = []

        if doc_type:
            conditions.append("type = ?")
            params.append(doc_type)
        if project:
            conditions.append("project = ?")
            params.append(project)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        rows = self._db.execute(
            f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def count_by_type(self) -> dict[str, int]:
        """Count documents grouped by type."""
        rows = self._db.execute(
            "SELECT type, COUNT(*) as count FROM documents GROUP BY type"
        ).fetchall()
        return {r["type"]: r["count"] for r in rows}

    def total_count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) as c FROM documents").fetchone()
        return row["c"]

    # -------------------------------------------------------------------------
    # FTS5 Search
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        project: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Full-text search with BM25 ranking.

        Title matches weighted 10x, tag matches 5x, content 1x.

        Returns:
            List of matching documents sorted by relevance.
        """
        if not query or not query.strip():
            return []

        # Escape FTS5 special characters for safe querying
        safe_query = self._escape_fts_query(query)
        if not safe_query:
            return []

        conditions = []
        params: list[Any] = [safe_query]

        if doc_type:
            conditions.append("d.type = ?")
            params.append(doc_type)
        if project:
            conditions.append("d.project = ?")
            params.append(project)

        extra_where = f"AND {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        sql = f"""
        SELECT d.*, bm25(documents_fts, 10.0, 1.0, 5.0) AS rank
        FROM documents_fts fts
        JOIN documents d ON d.rowid = fts.rowid
        WHERE documents_fts MATCH ?
        {extra_where}
        ORDER BY rank
        LIMIT ?
        """
        try:
            rows = self._db.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS query — return empty
            return []

        return [dict(r) for r in rows]

    @staticmethod
    def _escape_fts_query(query: str) -> str:
        """Escape a query for safe FTS5 MATCH usage.

        Wraps each token in double quotes to treat as literal.
        """
        tokens = query.strip().split()
        if not tokens:
            return ""
        # Quote each token to prevent FTS5 syntax interpretation
        return " ".join(f'"{t}"' for t in tokens)

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    def store_embedding(
        self, *, doc_id: str, embedding: bytes, model: str, dimensions: int
    ) -> None:
        """Store an embedding vector for a document."""
        now = datetime.now(UTC).isoformat()
        self._db.execute(
            """INSERT OR REPLACE INTO embeddings (doc_id, embedding, model, dimensions, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (doc_id, embedding, model, dimensions, now),
        )
        self._db.commit()

    def get_embedding(self, doc_id: str) -> bytes | None:
        """Get the raw embedding bytes for a document."""
        row = self._db.execute(
            "SELECT embedding FROM embeddings WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row["embedding"] if row else None

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------

    def set_config(self, key: str, value: str) -> None:
        now = datetime.now(UTC).isoformat()
        self._db.execute(
            "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        self._db.commit()

    def get_config(self, key: str, default: str = "") -> str:
        row = self._db.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # -------------------------------------------------------------------------
    # Query Log
    # -------------------------------------------------------------------------

    def log_query(
        self,
        *,
        tool: str,
        params: dict[str, Any],
        result_ids: list[str],
        duration_ms: float,
        session_id: str = "",
    ) -> None:
        """Log a query for the learning loop."""
        now = datetime.now(UTC).isoformat()
        self._db.execute(
            """INSERT INTO query_log
               (timestamp, tool, params, result_ids, result_count, duration_ms, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (now, tool, json.dumps(params), json.dumps(result_ids),
             len(result_ids), duration_ms, session_id),
        )
        self._db.commit()

    def get_query_log(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
