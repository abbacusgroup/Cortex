"""Graph queries — structured traversals of the knowledge graph.

Provides high-level query patterns:
- Causal chain: follow causedBy/ledTo edges
- Contradiction map: all contradicts edges for a scope
- Entity neighborhood: everything connected to entity X
- Evolution timeline: supersedes chain for a decision
- Project overview: all objects + entities + edges for a project
"""

from __future__ import annotations

from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store

logger = get_logger("retrieval.graph")


class GraphQueries:
    """High-level graph query patterns."""

    def __init__(self, store: Store):
        self.store = store

    def causal_chain(self, obj_id: str, max_depth: int = 10) -> list[dict[str, Any]]:
        """Follow causedBy/ledTo edges to build a causal narrative.

        Returns:
            Ordered list of objects in the causal chain.
        """
        chain = []
        visited: set[str] = set()
        self._traverse_causal(obj_id, chain, visited, max_depth, "backward")
        chain.reverse()
        # Add the starting object
        start = self.store.content.get(obj_id)
        if start:
            chain.append(self._summarize(start))
        # Forward from starting object
        self._traverse_causal(obj_id, chain, visited, max_depth, "forward")
        return chain

    def _traverse_causal(
        self,
        obj_id: str,
        chain: list[dict[str, Any]],
        visited: set[str],
        remaining: int,
        direction: str,
    ) -> None:
        if remaining <= 0 or obj_id in visited:
            return
        visited.add(obj_id)

        rels = self.store.get_relationships(obj_id)
        for rel in rels:
            is_backward = (
                direction == "backward"
                and rel["rel_type"] == "causedBy"
                and rel["direction"] == "outgoing"
            )
            is_forward = (
                direction == "forward"
                and rel["rel_type"] == "ledTo"
                and rel["direction"] == "outgoing"
            )
            if is_backward or is_forward:
                other = self.store.content.get(rel["other_id"])
                if other and rel["other_id"] not in visited:
                    chain.append(self._summarize(other))
                    self._traverse_causal(
                        rel["other_id"], chain, visited, remaining - 1, direction
                    )

    def contradiction_map(
        self, scope: str | None = None
    ) -> list[dict[str, Any]]:
        """Get all contradiction edges, optionally scoped to a project.

        Returns:
            List of {object_a, object_b, title_a, title_b} pairs.
        """
        contradictions = []
        seen: set[tuple[str, str]] = set()

        objects = self.store.list_objects(
            obj_type=None, project=scope, limit=500
        )
        for obj in objects:
            obj_id = obj.get("id", "")
            rels = self.store.get_relationships(obj_id)
            for rel in rels:
                if rel["rel_type"] == "contradicts":
                    pair = tuple(sorted([obj_id, rel["other_id"]]))
                    if pair not in seen:
                        seen.add(pair)
                        other = self.store.content.get(rel["other_id"])
                        contradictions.append({
                            "object_a": pair[0],
                            "object_b": pair[1],
                            "title_a": obj.get("title", ""),
                            "title_b": other.get("title", "") if other else "",
                        })
        return contradictions

    def entity_neighborhood(
        self, entity_name: str, *, max_hops: int = 2
    ) -> dict[str, Any]:
        """Get everything connected to an entity within N hops.

        Returns:
            Dict with entity info, direct objects, and extended neighborhood.
        """
        # Find entity
        entities = self.store.graph.list_entities()
        entity = None
        for e in entities:
            if e["name"].lower() == entity_name.lower():
                entity = e
                break

        if entity is None:
            return {"entity": None, "objects": [], "connections": []}

        # Get direct mentions (hop 1)
        mention_ids = self.store.graph.get_entity_mentions(entity["id"])
        direct_objects = []
        for mid in mention_ids:
            doc = self.store.content.get(mid)
            if doc:
                direct_objects.append(self._summarize(doc))

        # Extended neighborhood (hop 2): objects connected to the direct objects
        connections = []
        if max_hops >= 2:
            visited = set(mention_ids)
            for mid in mention_ids:
                rels = self.store.get_relationships(mid)
                for rel in rels:
                    other_id = rel["other_id"]
                    if other_id not in visited:
                        visited.add(other_id)
                        other = self.store.content.get(other_id)
                        if other:
                            connections.append({
                                **self._summarize(other),
                                "via": mid,
                                "via_rel": rel["rel_type"],
                            })

        return {
            "entity": entity,
            "objects": direct_objects,
            "connections": connections,
        }

    def evolution_timeline(self, obj_id: str) -> list[dict[str, Any]]:
        """Follow supersedes chain to show evolution of a decision/object.

        Returns:
            Chronologically ordered list (oldest first).
        """
        chain = []
        visited: set[str] = set()

        # Go backward (find what this superseded)
        self._traverse_supersedes(obj_id, chain, visited, "backward")
        chain.reverse()

        # Add current
        current = self.store.content.get(obj_id)
        if current:
            chain.append(self._summarize(current))

        # Go forward (find what supersedes this)
        self._traverse_supersedes(obj_id, chain, visited, "forward")

        return chain

    def _traverse_supersedes(
        self,
        obj_id: str,
        chain: list[dict[str, Any]],
        visited: set[str],
        direction: str,
    ) -> None:
        if obj_id in visited:
            return
        visited.add(obj_id)

        rels = self.store.get_relationships(obj_id)
        for rel in rels:
            if direction == "backward":
                if (
                    rel["rel_type"] == "supersedes"
                    and rel["direction"] == "outgoing"
                    and rel["other_id"] not in visited
                ):
                    other = self.store.content.get(rel["other_id"])
                    if other:
                        chain.append(self._summarize(other))
                        self._traverse_supersedes(
                            rel["other_id"], chain, visited, direction
                        )
            elif (
                direction == "forward"
                and rel["rel_type"] == "supersedes"
                and rel["direction"] == "incoming"
                and rel["other_id"] not in visited
            ):
                other = self.store.content.get(rel["other_id"])
                if other:
                    chain.append(self._summarize(other))
                    self._traverse_supersedes(
                        rel["other_id"], chain, visited, direction
                    )

    def project_overview(self, project: str) -> dict[str, Any]:
        """Get all objects, entities, and relationships for a project.

        Returns:
            Dict with objects, entities, and edges.
        """
        objects = self.store.list_objects(project=project, limit=500)

        # Collect all relationships
        edges = []
        for obj in objects:
            obj_id = obj.get("id", "")
            rels = self.store.get_relationships(obj_id)
            for rel in rels:
                if rel["direction"] == "outgoing":
                    edges.append({
                        "from": obj_id,
                        "to": rel["other_id"],
                        "type": rel["rel_type"],
                    })

        # Collect entities mentioned by project objects
        obj_ids = {obj.get("id", "") for obj in objects}
        entity_ids: set[str] = set()
        for entity in self.store.graph.list_entities():
            mentions = self.store.graph.get_entity_mentions(entity["id"])
            if obj_ids & set(mentions):
                entity_ids.add(entity["id"])

        all_entities = self.store.graph.list_entities()
        entities = [e for e in all_entities if e["id"] in entity_ids]

        return {
            "project": project,
            "object_count": len(objects),
            "objects": [self._summarize(obj) for obj in objects],
            "entities": entities,
            "edges": edges,
        }

    @staticmethod
    def _summarize(doc: dict[str, Any]) -> dict[str, Any]:
        """Create a compact summary of a document."""
        return {
            "id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "type": doc.get("type", ""),
            "project": doc.get("project", ""),
            "created_at": doc.get("created_at", ""),
        }
