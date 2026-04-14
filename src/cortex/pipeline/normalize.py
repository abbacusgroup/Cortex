"""Normalize stage — classify content, extract entities, generate embeddings.

Takes a raw capture and produces a structured knowledge object with:
- Type classification (via LLM or fallback)
- Summary and tag extraction
- Entity extraction
- Embedding generation
"""

from __future__ import annotations

import struct
from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.services.embeddings import EmbeddingProvider
from cortex.services.llm import LLMClient

logger = get_logger("pipeline.normalize")


class NormalizeStage:
    """Classify and enrich a raw knowledge object."""

    def __init__(
        self,
        store: Store,
        llm: LLMClient,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.store = store
        self.llm = llm
        self._embedding_provider = embedding_provider

    def run(self, obj_id: str) -> dict[str, Any]:
        """Run normalization on a knowledge object.

        Args:
            obj_id: The object ID to normalize.

        Returns:
            Dict with classification results and extracted entities.
        """
        doc = self.store.content.get(obj_id)
        if doc is None:
            logger.warning("Object not found for normalization: %s", obj_id)
            return {"status": "not_found"}

        title = doc.get("title", "")
        content = doc.get("content", "")
        existing_summary = doc.get("summary", "")

        # Step 1: Classification — skip LLM if pre-classified at capture time
        if existing_summary:
            # Pre-classified at capture time — skip LLM, use stored data
            classification = {
                "type": doc.get("type", "idea"),
                "summary": existing_summary,
                "tags": doc.get("tags", ""),
                "project": doc.get("project", ""),
                "entities": [],
                "confidence": float(doc.get("confidence", 0.0)),
                "properties": {},
            }
            logger.debug("Skipping LLM classify for %s — pre-classified", obj_id)
        else:
            # No pre-classification — use LLM or fallback
            classification = self.llm.classify(title=title, content=content)
            logger.debug(
                "Classified %s as %s (%.2f)",
                obj_id,
                classification["type"],
                classification["confidence"],
            )

        # Step 2: Update the object with classification results
        # Only override type if the classification has meaningful confidence;
        # fallback (confidence=0) means no LLM was available, so keep original.
        updates: dict[str, Any] = {
            "summary": classification["summary"],
            "confidence": classification["confidence"],
            "pipeline_stage": "normalized",
        }
        if classification["confidence"] > 0.0:
            updates["type"] = classification["type"]
        if classification["tags"] and not doc.get("tags"):
            updates["tags"] = classification["tags"]
        if classification["project"] and not doc.get("project"):
            updates["project"] = classification["project"]

        # Update in both stores
        self.store.content.update(obj_id, **updates)

        # Update graph store type-specific properties
        graph_updates = {"summary": classification["summary"]}
        if classification.get("properties"):
            graph_updates.update(
                {k: v for k, v in classification["properties"].items() if isinstance(v, str)}
            )
        try:
            self.store.graph.update_object(obj_id, **graph_updates)
        except Exception as e:
            logger.warning("Graph update failed during normalization for %s: %s", obj_id, e)

        # Step 3: Generate embedding
        self._generate_embedding(obj_id, title, content)

        return {
            "status": "normalized",
            "type": classification["type"],
            "confidence": classification["confidence"],
            "entities": classification["entities"],
            "properties": classification.get("properties", {}),
        }

    def _generate_embedding(self, obj_id: str, title: str, content: str) -> None:
        """Generate and store embedding for the object."""
        if self._embedding_provider is None:
            return

        try:
            text = f"{title}\n{content[:2000]}"
            vector = self._embedding_provider.embed(text)
            if vector is None:
                return
            embedding_bytes = struct.pack(f"{len(vector)}f", *vector)

            self.store.content.store_embedding(
                doc_id=obj_id,
                embedding=embedding_bytes,
                model=self._embedding_provider.model_name,
                dimensions=len(vector),
            )
            logger.debug("Stored embedding for %s (%d dims)", obj_id, len(vector))
        except Exception as e:
            logger.warning("Embedding generation failed for %s: %s", obj_id, e)
