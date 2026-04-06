"""Unified store — coordinates Oxigraph and SQLite for dual-write consistency.

All mutations go through this layer to keep both stores in sync.
Reads are routed to the appropriate store based on query type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cortex.core.config import CortexConfig
from cortex.core.errors import NotFoundError, SyncError
from cortex.core.logging import get_logger
from cortex.db.content_store import ContentStore
from cortex.db.graph_store import GraphStore

logger = get_logger("db.store")


class Store:
    """Unified store coordinating Oxigraph (graph) and SQLite (content)."""

    def __init__(self, config: CortexConfig):
        self.config = config
        self.graph = GraphStore(path=config.graph_db_path)
        self.content = ContentStore(path=config.sqlite_db_path)
        self._initialized = False
        self.temporal = None

    def initialize(self, ontology_path: Path | None = None) -> None:
        """Load ontology and mark store as ready."""
        self.graph.load_ontology(ontology_path)
        self._initialized = True
        from cortex.pipeline.temporal import TemporalVersioning
        self.temporal = TemporalVersioning(self.content)
        logger.info("Store initialized")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def close(self) -> None:
        self.content.close()

    # -------------------------------------------------------------------------
    # Knowledge Object CRUD (dual-write)
    # -------------------------------------------------------------------------

    def create(
        self,
        *,
        obj_type: str,
        title: str,
        content: str = "",
        raw_markdown: str = "",
        properties: dict[str, str] | None = None,
        project: str = "",
        tags: str = "",
        summary: str = "",
        tier: str = "archive",
        captured_by: str = "",
        confidence: float = 1.0,
    ) -> str:
        """Create a knowledge object in both stores.

        Returns:
            Object ID.

        Raises:
            SyncError: If one store succeeds but the other fails.
        """
        # Write to graph first (generates the ID)
        try:
            obj_id = self.graph.create_object(
                obj_type=obj_type,
                title=title,
                content=content,
                properties=properties,
                project=project,
                tags=tags,
                tier=tier,
                captured_by=captured_by,
                confidence=confidence,
            )
        except Exception as e:
            raise SyncError("Graph store write failed", cause=e)

        # Write to SQLite
        try:
            self.content.insert(
                doc_id=obj_id,
                title=title,
                content=content,
                raw_markdown=raw_markdown or content,
                doc_type=obj_type,
                project=project,
                tags=tags,
                summary=summary,
                tier=tier,
                captured_by=captured_by,
                confidence=confidence,
            )
        except Exception as e:
            # Rollback graph on SQLite failure
            self.graph.delete_object(obj_id)
            raise SyncError("SQLite write failed, graph rolled back", cause=e)

        logger.debug("Created %s in both stores (type=%s)", obj_id, obj_type)
        return obj_id

    def read(self, obj_id: str) -> dict[str, Any] | None:
        """Read from SQLite (content-authoritative) enriched with graph relationships."""
        doc = self.content.get(obj_id)
        if doc is None:
            return None

        # Enrich with relationships from graph
        doc["relationships"] = self.graph.get_relationships(obj_id)
        return doc

    def update(self, obj_id: str, **updates: Any) -> bool:
        """Update in both stores.

        Raises:
            NotFoundError: If object doesn't exist.
            SyncError: If stores go out of sync.
        """
        if self.temporal is not None:
            self.temporal.snapshot_before_update(obj_id)
        # Update SQLite
        try:
            self.content.update(obj_id, **updates)
        except NotFoundError:
            raise
        except Exception as e:
            raise SyncError("SQLite update failed", cause=e)

        # Update graph (only string-valued properties)
        graph_updates = {k: str(v) for k, v in updates.items() if isinstance(v, str)}
        if graph_updates:
            try:
                self.graph.update_object(obj_id, **graph_updates)
            except NotFoundError:
                pass  # Graph might not have all properties — OK
            except Exception as e:
                logger.warning("Graph update failed for %s: %s", obj_id, e)

        return True

    def delete(self, obj_id: str) -> bool:
        """Delete from both stores.

        Returns:
            True if deleted from at least one store.
        """
        graph_deleted = self.graph.delete_object(obj_id)
        content_deleted = self.content.delete(obj_id)
        return graph_deleted or content_deleted

    # -------------------------------------------------------------------------
    # Search & List (routed to appropriate store)
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        project: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Full-text search via SQLite FTS5."""
        return self.content.search(query, doc_type=doc_type, project=project, limit=limit)

    def list_objects(
        self,
        *,
        obj_type: str | None = None,
        project: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List objects via SQLite (has all metadata)."""
        return self.content.list_documents(
            doc_type=obj_type, project=project, limit=limit, offset=offset
        )

    # -------------------------------------------------------------------------
    # Relationships (graph-only)
    # -------------------------------------------------------------------------

    def create_relationship(
        self, *, from_id: str, rel_type: str, to_id: str, **kwargs: Any
    ) -> bool:
        return self.graph.create_relationship(
            from_id=from_id, rel_type=rel_type, to_id=to_id, **kwargs
        )

    def delete_relationship(self, *, from_id: str, rel_type: str, to_id: str) -> bool:
        return self.graph.delete_relationship(
            from_id=from_id, rel_type=rel_type, to_id=to_id
        )

    def get_relationships(self, obj_id: str) -> list[dict[str, str]]:
        return self.graph.get_relationships(obj_id)

    # -------------------------------------------------------------------------
    # Entities (graph-only)
    # -------------------------------------------------------------------------

    def create_entity(self, *, name: str, entity_type: str = "concept", aliases: str = "") -> str:
        return self.graph.create_entity(name=name, entity_type=entity_type, aliases=aliases)

    def add_mention(self, *, obj_id: str, entity_id: str) -> None:
        self.graph.add_mention(obj_id=obj_id, entity_id=entity_id)

    def list_entities(self, entity_type: str | None = None) -> list[dict[str, str]]:
        return self.graph.list_entities(entity_type=entity_type)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Get store health and counts."""
        content_counts = self.content.count_by_type()
        graph_counts = self.graph.count_by_type()
        return {
            "initialized": self._initialized,
            "sqlite_total": self.content.total_count(),
            "graph_triples": self.graph.triple_count,
            "counts_by_type": content_counts,
            "graph_counts_by_type": graph_counts,
            "entities": len(self.graph.list_entities()),
        }
