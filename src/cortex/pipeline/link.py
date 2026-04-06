"""Link stage — entity resolution and relationship discovery.

Takes normalized entities and discovers connections to the knowledge graph.
"""

from __future__ import annotations

from typing import Any

from cortex.core.constants import RELATIONSHIP_TYPES
from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.services.llm import LLMClient

logger = get_logger("pipeline.link")


class LinkStage:
    """Resolve entities and discover relationships."""

    def __init__(self, store: Store, llm: LLMClient):
        self.store = store
        self.llm = llm

    def run(
        self,
        obj_id: str,
        entities: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run linking on a knowledge object.

        Args:
            obj_id: The object ID to link.
            entities: List of extracted entities [{name, type}].

        Returns:
            Dict with resolved entities and discovered relationships.
        """
        # Step 1: Resolve entities
        resolved = self._resolve_entities(obj_id, entities)

        # Step 2: Discover relationships via LLM
        relationships = self._discover_relationships(obj_id)

        # Step 3: Update pipeline stage
        try:
            self.store.content.update(obj_id, pipeline_stage="linked")
        except Exception:
            pass

        return {
            "status": "linked",
            "entities_resolved": len(resolved),
            "relationships_created": len(relationships),
            "entities": resolved,
            "relationships": relationships,
        }

    def _resolve_entities(
        self, obj_id: str, entities: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Resolve extracted entity names against the graph.

        For each entity:
        1. Check if entity exists (case-insensitive match)
        2. If not, create a new entity node
        3. Add a mentions edge from the object to the entity
        """
        resolved = []
        for entity_info in entities:
            name = entity_info.get("name", "").strip()
            if not name:
                continue

            entity_type = entity_info.get("type", "concept")

            # Create or get entity (handles dedup internally)
            entity_id = self.store.create_entity(
                name=name, entity_type=entity_type
            )

            # Add mentions edge
            self.store.add_mention(obj_id=obj_id, entity_id=entity_id)

            resolved.append({
                "name": name,
                "type": entity_type,
                "entity_id": entity_id,
                "action": "existing" if entity_id else "created",
            })
            logger.debug("Resolved entity '%s' → %s", name, entity_id)

        return resolved

    def _discover_relationships(self, obj_id: str) -> list[dict[str, Any]]:
        """Use LLM to discover relationships with existing objects."""
        doc = self.store.content.get(obj_id)
        if doc is None:
            return []

        # Get recent objects to compare against
        recent = self.store.list_objects(limit=30)
        # Exclude self
        candidates = [r for r in recent if r.get("id") != obj_id]
        if not candidates:
            return []

        # Ask LLM for relationships
        discovered = self.llm.discover_relationships(
            new_id=obj_id,
            new_title=doc.get("title", ""),
            new_type=doc.get("type", "idea"),
            new_content=doc.get("content", ""),
            existing=candidates,
        )

        # Validate and create relationships
        created = []
        for rel in discovered:
            rel_type = rel.get("rel_type", "")
            if rel_type not in RELATIONSHIP_TYPES:
                continue

            from_id = rel.get("from_id", "")
            to_id = rel.get("to_id", "")
            if not from_id or not to_id:
                continue
            if from_id == to_id:
                continue

            # Verify both objects exist
            if not self.store.content.get(from_id) or not self.store.content.get(to_id):
                logger.debug("Skipping relationship — object not found")
                continue

            try:
                self.store.create_relationship(
                    from_id=from_id,
                    rel_type=rel_type,
                    to_id=to_id,
                )
                created.append(rel)
                logger.debug(
                    "Created relationship: %s -[%s]-> %s",
                    from_id[:8], rel_type, to_id[:8],
                )
            except Exception as e:
                logger.warning("Failed to create relationship: %s", e)

        return created
