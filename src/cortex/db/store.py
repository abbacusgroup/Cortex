"""Unified store — coordinates Oxigraph and SQLite for dual-write consistency.

All mutations go through this layer to keep both stores in sync.
Reads are routed to the appropriate store based on query type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cortex.core.config import CortexConfig
from cortex.core.constants import KNOWLEDGE_TYPES
from cortex.core.errors import NotFoundError, SyncError, ValidationError
from cortex.core.logging import get_logger
from cortex.db.content_store import ContentStore
from cortex.db.graph_store import GraphStore

logger = get_logger("db.store")

# SQLite column name → graph predicate local name. Only these columns have a
# graph representation; the mapping fixes the historical bug where snake_case
# column names (captured_by) were forwarded verbatim as graph predicates,
# diverging from the camelCase predicates written at creation (capturedBy).
# ``type`` is handled separately because it maps to the object's rdf:type
# class assertion, not a literal predicate.
_COLUMN_TO_PREDICATE: dict[str, str] = {
    "title": "title",
    "content": "content",
    "project": "project",
    "tags": "tags",
    "summary": "summary",
    "tier": "tier",
    "captured_by": "capturedBy",
}

# SQLite-only columns, intentionally never mirrored to the graph:
# raw_markdown / pipeline_stage / updated_at have no graph representation;
# confidence is written to the graph as a typed xsd:float literal at creation
# time and update_object would rewrite it as a plain string literal, so it
# stays content-authoritative on update.
_CONTENT_ONLY_COLUMNS: frozenset[str] = frozenset(
    {"raw_markdown", "pipeline_stage", "updated_at", "confidence"}
)


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

    def close(self) -> None:
        self.content.close()
        self.graph.close()

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
        created_at: str | None = None,
        updated_at: str | None = None,
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
                captured_at=created_at,
            )
        except Exception as e:
            raise SyncError("Graph store write failed", cause=e) from e

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
                created_at=created_at,
                updated_at=updated_at,
            )
        except Exception as e:
            # Rollback graph on SQLite failure
            self.graph.delete_object(obj_id)
            raise SyncError("SQLite write failed, graph rolled back", cause=e) from e

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

    def update(
        self,
        obj_id: str,
        *,
        properties: dict[str, str] | None = None,
        **updates: Any,
    ) -> bool:
        """Update in both stores — the single dual-write update path.

        SQLite column updates are mirrored to the graph using the explicit
        column→predicate mapping (``captured_by`` → ``capturedBy`` etc.);
        SQLite-only columns (``raw_markdown``, ``pipeline_stage``, ...) are
        never forwarded. A ``type`` change rewrites the object's rdf:type
        class assertion in the graph so reclassification keeps both stores'
        per-type counts consistent.

        Args:
            obj_id: Object ID.
            properties: Optional graph-only, type-specific properties
                (e.g. rationale, symptom) written as cortex predicates —
                the update-time counterpart of ``create(properties=...)``.
                Non-string values are dropped.
            **updates: SQLite column updates.

        Raises:
            NotFoundError: If object doesn't exist.
            ValidationError: If ``type`` is given but isn't a valid knowledge
                type (raised before either store is touched).
            SyncError: If stores go out of sync.
        """
        # Validate type up front so an invalid value can't reach SQLite and
        # leave the stores diverged when the graph later rejects it.
        if "type" in updates and updates["type"] not in KNOWLEDGE_TYPES:
            raise ValidationError(
                f"Invalid knowledge type: {updates['type']}",
                context={"type": updates["type"], "valid_types": sorted(KNOWLEDGE_TYPES)},
            )

        graph_updates = self._graph_updates_for(obj_id, updates, properties)

        if self.temporal is not None:
            self.temporal.snapshot_before_update(obj_id)
        # Update SQLite
        try:
            self.content.update(obj_id, **updates)
        except NotFoundError:
            raise
        except Exception as e:
            raise SyncError("SQLite update failed", cause=e) from e

        # Update graph
        if graph_updates:
            try:
                self.graph.update_object(obj_id, **graph_updates)
            except NotFoundError:
                # Object node is absent from the graph store entirely, so there
                # is nothing to update there. SQLite already succeeded; we keep
                # going but flag the divergence between the two stores.
                logger.warning(
                    "Object %s missing from graph during update — stores may be out of sync",
                    obj_id,
                )
            except Exception as e:
                logger.warning(
                    "Graph update failed for %s (SQLite update succeeded): %s",
                    obj_id, e,
                )
                raise SyncError(
                    f"SQLite updated but graph update failed for {obj_id}",
                    cause=e,
                ) from e

        return True

    @staticmethod
    def _graph_updates_for(
        obj_id: str,
        updates: dict[str, Any],
        properties: dict[str, str] | None,
    ) -> dict[str, str]:
        """Translate SQLite column updates into graph predicate updates.

        Columns without a graph representation are dropped; mapped columns
        are renamed to their ontology predicate; ``type`` is forwarded for
        the rdf:type class rewrite. Graph-only ``properties`` are merged on
        top (they may not carry ``type`` — the class is set via the column).
        """
        graph_updates: dict[str, str] = {}
        for key, value in updates.items():
            if key in _CONTENT_ONLY_COLUMNS:
                continue
            if key == "type":
                graph_updates["type"] = str(value)
            elif key in _COLUMN_TO_PREDICATE:
                graph_updates[_COLUMN_TO_PREDICATE[key]] = "" if value is None else str(value)
            # Unknown columns are left out; ContentStore.update will reject
            # them before any write happens.
        if properties:
            for key, value in properties.items():
                if key == "type":
                    logger.warning(
                        "Ignoring 'type' in graph properties for %s — "
                        "pass type as a column update instead",
                        obj_id,
                    )
                    continue
                if isinstance(value, str):
                    graph_updates[key] = value
        return graph_updates

    def delete(self, obj_id: str) -> bool:
        """Delete from both stores.

        The current SQLite row is snapshotted into the temporal version
        history first, so a deletion is recoverable — mirroring update().

        Returns:
            True if deleted from at least one store.

        Raises:
            SyncError: If either store's delete fails. The message names
                which store succeeded so the direction of drift is known.
        """
        if self.temporal is not None:
            self.temporal.snapshot_before_update(obj_id)

        try:
            graph_deleted = self.graph.delete_object(obj_id)
        except Exception as e:
            raise SyncError(
                f"Graph delete failed for {obj_id} (SQLite row not touched)",
                cause=e,
            ) from e

        try:
            content_deleted = self.content.delete(obj_id)
        except Exception as e:
            raise SyncError(
                f"Graph delete succeeded but SQLite delete failed for {obj_id}",
                cause=e,
            ) from e

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
        return self.graph.delete_relationship(from_id=from_id, rel_type=rel_type, to_id=to_id)

    def get_relationships(self, obj_id: str) -> list[dict[str, str]]:
        return self.graph.get_relationships(obj_id)

    # -------------------------------------------------------------------------
    # Entities (graph-only)
    # -------------------------------------------------------------------------

    def create_entity(
        self, *, name: str, entity_type: str = "concept", aliases: str = ""
    ) -> tuple[str, bool]:
        return self.graph.create_entity(name=name, entity_type=entity_type, aliases=aliases)

    def add_mention(self, *, obj_id: str, entity_id: str) -> None:
        self.graph.add_mention(obj_id=obj_id, entity_id=entity_id)

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from the graph."""
        return self.graph.delete_entity(entity_id)

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
            "entities": self.graph.count_entities(),
        }
