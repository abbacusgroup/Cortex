"""Import/migration — bring data from external sources into Cortex v2.

Supports:
- Cortex v1 SQLite database
- Obsidian vault (markdown files with YAML frontmatter)
- Deduplication via content hash matching
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store

logger = get_logger("pipeline.importer")


class CortexV1Importer:
    """Import from Cortex v1 SQLite database."""

    def __init__(self, store: Store):
        self.store = store

    def run(self, v1_db_path: Path) -> dict[str, Any]:
        """Import all documents from a v1 database.

        Args:
            v1_db_path: Path to the v1 SQLite database.

        Returns:
            Dict with imported, skipped (duplicates), and failed counts.
        """
        if not v1_db_path.exists():
            return {"status": "error", "message": f"File not found: {v1_db_path}"}

        imported = 0
        skipped = 0
        failed = 0

        try:
            v1_db = sqlite3.connect(str(v1_db_path))
            v1_db.row_factory = sqlite3.Row
        except Exception as e:
            return {"status": "error", "message": str(e)}

        try:
            rows = v1_db.execute("SELECT * FROM documents ORDER BY created_at ASC").fetchall()
        except sqlite3.OperationalError:
            # Try alternative table names
            try:
                rows = v1_db.execute("SELECT * FROM notes ORDER BY created_at ASC").fetchall()
            except sqlite3.OperationalError:
                v1_db.close()
                return {"status": "error", "message": "No documents or notes table found"}

        for row in rows:
            row_dict = dict(row)
            try:
                content = row_dict.get("content", "") or ""
                title = row_dict.get("title", "") or row_dict.get("name", "") or "Untitled"

                # Dedup check
                if self._is_duplicate(title, content):
                    skipped += 1
                    continue

                obj_type = self._map_v1_type(row_dict.get("type", ""))

                self.store.create(
                    obj_type=obj_type,
                    title=title,
                    content=content,
                    raw_markdown=content,
                    project=row_dict.get("project", "") or "",
                    tags=row_dict.get("tags", "") or "",
                    captured_by="import-v1",
                )
                imported += 1
            except Exception as e:
                logger.warning("Failed to import v1 doc '%s': %s", row_dict.get("title", "?"), e)
                failed += 1

        v1_db.close()
        logger.info("V1 import: %d imported, %d skipped, %d failed", imported, skipped, failed)
        return {
            "status": "ok",
            "imported": imported,
            "skipped": skipped,
            "failed": failed,
            "total": len(rows),
        }

    @staticmethod
    def _map_v1_type(v1_type: str) -> str:
        """Map v1 type names to v2 ontology types."""
        mapping = {
            "capture": "idea",
            "note": "idea",
            "fix": "fix",
            "session": "session",
            "decision": "decision",
            "lesson": "lesson",
            "research": "research",
            "source": "source",
            "synthesis": "synthesis",
            "idea": "idea",
            "guide": "research",
            "workflow": "session",
        }
        return mapping.get(v1_type.lower(), "idea")

    def _is_duplicate(self, title: str, content: str) -> bool:
        """Check if content already exists via hash."""
        content_hash = hashlib.sha256(
            f"{title}:{content}".encode()
        ).hexdigest()

        existing = self.store.content.get_config(f"import_hash:{content_hash}", "")
        if existing:
            return True

        # Store hash for future dedup
        self.store.content.set_config(f"import_hash:{content_hash}", "1")
        return False


class ObsidianImporter:
    """Import from an Obsidian vault (markdown files)."""

    def __init__(self, store: Store):
        self.store = store

    def run(self, vault_path: Path) -> dict[str, Any]:
        """Import all markdown files from a vault.

        Args:
            vault_path: Path to the Obsidian vault root.

        Returns:
            Dict with imported, skipped, and failed counts.
        """
        if not vault_path.exists() or not vault_path.is_dir():
            return {"status": "error", "message": f"Not a directory: {vault_path}"}

        md_files = sorted(vault_path.rglob("*.md"))
        if not md_files:
            return {
                "status": "ok",
                "message": "Nothing to import — no .md files found",
                "imported": 0,
                "skipped": 0,
                "failed": 0,
                "total": 0,
            }

        imported = 0
        skipped = 0
        failed = 0

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                title = md_file.stem
                relative = str(md_file.relative_to(vault_path))

                # Dedup check
                if self._is_duplicate(title, content):
                    skipped += 1
                    continue

                # Parse frontmatter
                meta = self._parse_frontmatter(content)
                body = self._strip_frontmatter(content)

                obj_type = self._infer_type(meta, relative)
                project = meta.get("project", "") or self._infer_project(relative)
                tags = meta.get("tags", "")
                if isinstance(tags, list):
                    tags = ",".join(tags)

                self.store.create(
                    obj_type=obj_type,
                    title=title,
                    content=body,
                    raw_markdown=content,
                    project=project,
                    tags=tags,
                    captured_by="import-obsidian",
                )
                imported += 1
            except Exception as e:
                logger.warning("Failed to import '%s': %s", md_file.name, e)
                failed += 1

        logger.info(
            "Obsidian import: %d imported, %d skipped, %d failed",
            imported, skipped, failed,
        )
        return {
            "status": "ok",
            "imported": imported,
            "skipped": skipped,
            "failed": failed,
            "total": len(md_files),
        }

    def _is_duplicate(self, title: str, content: str) -> bool:
        content_hash = hashlib.sha256(
            f"{title}:{content}".encode()
        ).hexdigest()
        existing = self.store.content.get_config(f"import_hash:{content_hash}", "")
        if existing:
            return True
        self.store.content.set_config(f"import_hash:{content_hash}", "1")
        return False

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        """Parse YAML frontmatter from markdown."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return {}

        meta: dict[str, Any] = {}
        for line in match.group(1).split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                # Handle YAML lists (simple)
                if value.startswith("[") and value.endswith("]"):
                    value = [v.strip().strip("'\"") for v in value[1:-1].split(",")]
                meta[key] = value
        return meta

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """Remove YAML frontmatter from content."""
        return re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)

    @staticmethod
    def _infer_type(meta: dict[str, Any], relative_path: str) -> str:
        """Infer knowledge type from frontmatter or file path."""
        # Check frontmatter type
        ft = str(meta.get("type", "")).lower()
        valid = {"decision", "lesson", "fix", "session", "research", "source", "synthesis", "idea"}
        if ft in valid:
            return ft

        # Infer from directory
        path_lower = relative_path.lower()
        for t in valid:
            plural = t + "s"
            matches = (
                f"/{t}/" in path_lower
                or f"/{plural}/" in path_lower
                or path_lower.startswith(f"{t}/")
                or path_lower.startswith(f"{plural}/")
            )
            if matches:
                return t

        return "idea"

    @staticmethod
    def _infer_project(relative_path: str) -> str:
        """Infer project from directory structure."""
        parts = Path(relative_path).parts
        if len(parts) > 1:
            return parts[0]
        return ""
