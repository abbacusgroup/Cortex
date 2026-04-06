"""Enrich stage — tier assignment, staleness scoring, connection counting.

Determines the memory tier (archive/recall/reflex) and computes
metadata about the object's position in the knowledge graph.
"""

from __future__ import annotations

from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store

logger = get_logger("pipeline.enrich")


class EnrichStage:
    """Compute tier, staleness, and connection metadata."""

    def __init__(self, store: Store):
        self.store = store

    def run(self, obj_id: str) -> dict[str, Any]:
        """Run enrichment on a knowledge object.

        Returns:
            Dict with tier, connection_count, staleness_score.
        """
        doc = self.store.content.get(obj_id)
        if doc is None:
            return {"status": "not_found"}

        # Compute tier
        tier = self._compute_tier(doc)

        # Count connections
        relationships = self.store.get_relationships(obj_id)
        connection_count = len(relationships)

        # Compute staleness
        staleness = self._compute_staleness(obj_id, relationships)

        # Update the object
        updates: dict[str, Any] = {
            "tier": tier,
            "pipeline_stage": "enriched",
        }
        try:
            self.store.content.update(obj_id, **updates)
            self.store.graph.update_object(obj_id, tier=tier)
        except Exception as e:
            logger.warning("Failed to update enrichment for %s: %s", obj_id, e)

        return {
            "status": "enriched",
            "tier": tier,
            "connection_count": connection_count,
            "staleness_score": staleness,
        }

    def _compute_tier(self, doc: dict[str, Any]) -> str:
        """Determine memory tier based on classification state.

        Rules:
        - No summary or confidence=0 → archive
        - Has summary and confidence > 0 → recall
        - Explicitly promoted or high access → reflex (handled by learning loop)
        """
        confidence = doc.get("confidence", 0.0)
        summary = doc.get("summary", "")
        current_tier = doc.get("tier", "archive")

        # Don't demote from reflex (that's the learning loop's job)
        if current_tier == "reflex":
            return "reflex"

        if summary and confidence > 0:
            return "recall"

        return "archive"

    def _compute_staleness(
        self, obj_id: str, relationships: list[dict[str, str]]
    ) -> float:
        """Compute a staleness score (0.0 = fresh, 1.0 = very stale).

        An object is stale if:
        - It depends on something that has been superseded
        - It has no recent connections
        """
        staleness = 0.0

        for rel in relationships:
            if rel["direction"] == "outgoing" and rel["rel_type"] == "dependsOn":
                # Check if the dependency has been superseded
                dep = self.store.content.get(rel["other_id"])
                if dep and dep.get("pipeline_stage") == "superseded":
                    staleness += 0.5

        return min(staleness, 1.0)

    def promote_to_reflex(self, obj_id: str) -> bool:
        """Explicitly promote an object to the reflex tier."""
        try:
            self.store.content.update(obj_id, tier="reflex")
            self.store.graph.update_object(obj_id, tier="reflex")
            return True
        except Exception:
            return False

    def demote_from_reflex(self, obj_id: str) -> bool:
        """Demote an object from reflex back to recall."""
        try:
            self.store.content.update(obj_id, tier="recall")
            self.store.graph.update_object(obj_id, tier="recall")
            return True
        except Exception:
            return False
