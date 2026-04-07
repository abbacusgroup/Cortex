"""Import/migration — bring data from external sources into Cortex v2.

Supports:
- Cortex v1 SQLite database
- Obsidian vault (markdown files with YAML frontmatter)
- Deduplication via content hash matching
"""

from __future__ import annotations

import difflib
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.pipeline.orchestrator import PipelineOrchestrator

logger = get_logger("pipeline.importer")

_TECHNOLOGIES = frozenset({
    "python", "javascript", "typescript", "rust", "go", "java",
    "react", "fastapi", "django", "flask", "express", "node",
    "docker", "kubernetes", "postgresql", "mysql", "sqlite", "redis",
    "mongodb", "neo4j", "elasticsearch", "kafka",
    "git", "github", "linux", "graphql", "pytorch", "tensorflow",
    "anthropic", "openai", "langchain", "oxigraph", "sparql",
    "nginx", "aws", "gcp", "azure", "terraform",
})


def _extract_entities_from_tags(store: Store, obj_id: str, tags: str) -> None:
    """Create entity nodes from tags and link to the document."""
    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
    for tag in tag_list:
        entity_type = "technology" if tag in _TECHNOLOGIES else "concept"
        entity_id = store.create_entity(name=tag, entity_type=entity_type)
        store.add_mention(obj_id=obj_id, entity_id=entity_id)


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

                tags = row_dict.get("tags", "") or ""
                obj_id = self.store.create(
                    obj_type=obj_type,
                    title=title,
                    content=content,
                    raw_markdown=content,
                    project=row_dict.get("project", "") or "",
                    tags=tags,
                    captured_by="import-v1",
                )
                if tags:
                    _extract_entities_from_tags(self.store, obj_id, tags)
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

        # Title-based fallback (direct SQL, not FTS)
        row = self.store.content._db.execute(
            "SELECT id FROM documents WHERE title = ? AND captured_by LIKE 'import-%' LIMIT 1",
            (title,),
        ).fetchone()
        if row:
            return True

        # Store hash for future dedup
        self.store.content.set_config(f"import_hash:{content_hash}", "1")
        return False


class ObsidianImporter:
    """Import from an Obsidian vault (markdown files)."""

    def __init__(self, store: Store, pipeline: PipelineOrchestrator | None = None):
        self.store = store
        self.pipeline = pipeline

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
        wiki_link_map: dict[str, list[str]] = {}

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                title = md_file.stem
                relative = str(md_file.relative_to(vault_path))

                # Parse frontmatter and strip before dedup
                meta = self._parse_frontmatter(content)
                body = self._strip_frontmatter(content)

                # Skip re-imported files (prefer originals)
                source = meta.get("source", "")
                if isinstance(source, str) and source.startswith("ingest:"):
                    skipped += 1
                    continue

                # Skip index files (organizational scaffolding)
                if str(meta.get("type", "")).lower() == "index":
                    skipped += 1
                    continue

                # Dedup check — content-only hash
                if self._is_duplicate(title, body):
                    skipped += 1
                    continue

                obj_type = self._infer_type(meta, relative)
                project = meta.get("project", "") or self._infer_project(relative)
                tags = meta.get("tags", "")
                if isinstance(tags, list):
                    tags = ",".join(tags)

                # Preserve timestamps, summary, confidence from frontmatter
                created_at = self._parse_date(meta.get("created", ""))
                updated_at = self._parse_date(meta.get("updated", ""))
                summary = meta.get("summary", "") if isinstance(meta.get("summary"), str) else ""
                # If summary is the block scalar indicator itself (>- etc.), clear it
                if summary.startswith(">") or summary.startswith("|"):
                    summary = ""
                fm_confidence = 0.85 if summary else 0.0

                # Extract wiki-links for second-pass relationship creation
                wiki_links = self._extract_wiki_links(body)

                # Build entities from tags + key_topics
                entities: list[dict[str, str]] = []
                tag_list = [
                    t.strip().lower()
                    for t in (tags.split(",") if isinstance(tags, str) else tags)
                    if t.strip()
                ]
                for tag in tag_list:
                    entity_type = "technology" if tag in _TECHNOLOGIES else "concept"
                    entities.append({"name": tag, "type": entity_type})

                key_topics = meta.get("key_topics", [])
                if isinstance(key_topics, list):
                    for topic in key_topics:
                        t = str(topic).strip().lower()
                        if t and not any(e["name"] == t for e in entities):
                            entities.append({"name": t, "type": "concept"})

                # Route through pipeline when available
                if self.pipeline:
                    result = self.pipeline.capture(
                        title=title,
                        content=body,
                        obj_type=obj_type,
                        project=project,
                        tags=tags if isinstance(tags, str) else ",".join(tags),
                        summary=summary,
                        captured_by="import-obsidian",
                        confidence=fm_confidence,
                        entities=entities if entities else None,
                        run_pipeline=True,
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                    obj_id = result["id"]
                else:
                    obj_id = self.store.create(
                        obj_type=obj_type,
                        title=title,
                        content=body,
                        raw_markdown=content,
                        project=project,
                        tags=tags if isinstance(tags, str) else ",".join(tags),
                        summary=summary,
                        confidence=fm_confidence,
                        captured_by="import-obsidian",
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                    if tags:
                        tag_str = tags if isinstance(tags, str) else ",".join(tags)
                        _extract_entities_from_tags(self.store, obj_id, tag_str)

                # Collect wiki-links for second pass
                if wiki_links:
                    wiki_link_map[obj_id] = wiki_links

                imported += 1
            except Exception as e:
                logger.warning("Failed to import '%s': %s", md_file.name, e)
                failed += 1

        # Second pass: resolve wiki-links to relationships
        links_created = self._resolve_wiki_links(wiki_link_map)

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
            "wiki_links_created": links_created,
        }

    def _is_duplicate(self, title: str, content: str) -> bool:
        """Check if content already exists via content-only hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = self.store.content.get_config(f"import_hash:{content_hash}", "")
        if existing:
            return True

        # Title-based fallback (direct SQL, not FTS)
        row = self.store.content._db.execute(
            "SELECT id FROM documents WHERE title = ? AND captured_by LIKE 'import-%' LIMIT 1",
            (title,),
        ).fetchone()
        if row:
            return True

        self.store.content.set_config(f"import_hash:{content_hash}", "1")
        return False

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        """Parse YAML frontmatter from markdown.

        Handles both inline lists ([a, b]) and multi-line lists:
            tags:
              - palantir
              - neo4j

        Also handles YAML block scalars:
            summary: >-
              Line one
              line two
        """
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return {}

        meta: dict[str, Any] = {}
        current_key: str | None = None
        current_list: list[str] | None = None
        current_scalar: str | None = None
        scalar_folded: bool = True

        for line in match.group(1).split("\n"):
            # Check for list item (indented "- value")
            list_match = re.match(r"^\s+- (.+)$", line)
            if list_match and current_key is not None and current_scalar is None:
                if current_list is None:
                    current_list = []
                current_list.append(list_match.group(1).strip().strip("'\""))
                continue

            # Accumulate block scalar continuation lines
            if current_scalar is not None and current_key is not None:
                if line.startswith("  ") or line.startswith("\t"):
                    current_scalar += line.strip() + "\n"
                    continue
                else:
                    # Scalar ended — save it
                    if scalar_folded:
                        meta[current_key] = " ".join(
                            current_scalar.strip().split("\n")
                        ).strip()
                    else:
                        meta[current_key] = current_scalar.strip()
                    current_scalar = None
                    current_key = None

            # Save any accumulated list
            if current_key is not None and current_list is not None:
                meta[current_key] = current_list
                current_list = None
                current_key = None

            # Check for key: value pair
            if ":" in line and not line.startswith(" "):
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()

                if not value:
                    # Value on next lines (multi-line list)
                    current_key = key
                    current_list = None
                    current_scalar = None
                elif value.startswith(">") or value.startswith("|"):
                    # Block scalar indicator
                    scalar_folded = value[0] == ">"
                    current_key = key
                    current_scalar = ""
                    current_list = None
                elif value.startswith("[") and value.endswith("]"):
                    # Inline list: [a, b, c]
                    meta[key] = [
                        v.strip().strip("'\"")
                        for v in value[1:-1].split(",")
                    ]
                else:
                    meta[key] = value.strip("'\"")
                    current_key = None

        # Save final accumulated scalar
        if current_scalar is not None and current_key is not None:
            if scalar_folded:
                meta[current_key] = " ".join(
                    current_scalar.strip().split("\n")
                ).strip()
            else:
                meta[current_key] = current_scalar.strip()

        # Save final accumulated list
        if current_key is not None and current_list is not None:
            meta[current_key] = current_list

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

    @staticmethod
    def _parse_date(date_str: str | Any) -> str | None:
        """Convert YYYY-MM-DD to ISO datetime string."""
        if not isinstance(date_str, str) or not date_str.strip():
            return None
        date_str = date_str.strip().strip("'\"")
        try:
            # Handle YYYY-MM-DD format
            if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                return f"{date_str}T00:00:00+00:00"
            return None
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _extract_wiki_links(content: str) -> list[str]:
        """Extract [[Title]] wiki-links from content."""
        matches = re.findall(r'\[\[([^\]]+)\]\]', content)
        return list(dict.fromkeys(matches))  # unique, preserving order

    def _resolve_wiki_links(self, wiki_link_map: dict[str, list[str]]) -> int:
        """Resolve wiki-links to 'supports' relationships (second pass)."""
        if not wiki_link_map:
            return 0

        # Build title -> id lookup
        all_docs = self.store.list_objects(limit=10000)
        title_index: dict[str, str] = {}
        normalized_index: dict[str, str] = {}
        for doc in all_docs:
            doc_title = doc.get("title", "")
            doc_id = doc.get("id", "")
            if doc_title and doc_id:
                title_index[doc_title] = doc_id
                # Normalize: lowercase, strip date prefix (YYYY-MM-DD-), strip special chars
                norm = re.sub(r'^\d{4}-\d{2}-\d{2}-?', '', doc_title.lower())
                norm = re.sub(r'[^a-z0-9\s]', '', norm).strip()
                normalized_index[norm] = doc_id

        created = 0
        for obj_id, targets in wiki_link_map.items():
            for target in targets:
                matched_id = None

                # Try exact match first
                if target in title_index:
                    matched_id = title_index[target]
                else:
                    # Try normalized fuzzy match
                    norm_target = re.sub(r'^\d{4}-\d{2}-\d{2}-?', '', target.lower())
                    norm_target = re.sub(r'[^a-z0-9\s]', '', norm_target).strip()

                    if norm_target in normalized_index:
                        matched_id = normalized_index[norm_target]
                    else:
                        # Fuzzy match with threshold
                        best_ratio = 0.0
                        best_id = None
                        for norm_title, doc_id in normalized_index.items():
                            ratio = difflib.SequenceMatcher(
                                None, norm_target, norm_title
                            ).ratio()
                            if ratio > best_ratio and ratio > 0.8:
                                best_ratio = ratio
                                best_id = doc_id
                        matched_id = best_id

                if matched_id and matched_id != obj_id:
                    try:
                        self.store.create_relationship(
                            from_id=obj_id,
                            rel_type="supports",
                            to_id=matched_id,
                        )
                        created += 1
                    except Exception as e:
                        logger.debug("Wiki-link relationship failed: %s", e)

        logger.info("Wiki-link resolution: %d relationships created", created)
        return created
