"""Hybrid retrieval engine — keyword + semantic + graph-aware ranking.

Combines FTS5 (BM25), embedding cosine similarity, and graph connectivity
into a single ranked result set with configurable weights.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.services.embeddings import EmbeddingProvider

logger = get_logger("retrieval.engine")

# Default ranking weights
DEFAULT_WEIGHTS = {
    "keyword": 0.4,
    "semantic": 0.3,
    "graph": 0.2,
    "recency": 0.1,
}


class RetrievalEngine:
    """Hybrid search combining keyword, semantic, graph, and recency signals."""

    def __init__(
        self,
        store: Store,
        weights: dict[str, float] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.store = store
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self._embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        project: str | None = None,
        entity: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Execute a hybrid search.

        Args:
            query: Search query text.
            doc_type: Filter by knowledge type.
            project: Filter by project.
            entity: Filter by entity name.
            limit: Maximum results to return.

        Returns:
            List of result dicts sorted by combined score (best first).
        """
        start = time.monotonic()

        if not query or not query.strip():
            return []

        # Gather candidates from multiple signals
        candidates: dict[str, dict[str, Any]] = {}

        # 1. Keyword search (FTS5)
        keyword_results = self.store.search(
            query, doc_type=doc_type, project=project, limit=limit * 3
        )
        for i, doc in enumerate(keyword_results):
            doc_id = doc["id"]
            if doc_id not in candidates:
                candidates[doc_id] = {**doc, "scores": {}}
            # Normalize rank: first result = 1.0, linearly decreasing
            candidates[doc_id]["scores"]["keyword"] = 1.0 - (i / max(len(keyword_results), 1))

        # 2. Semantic search (embedding similarity)
        semantic_results = self._semantic_search(query, limit=limit * 3)
        for doc_id, similarity in semantic_results:
            if doc_id not in candidates:
                doc = self.store.content.get(doc_id)
                if doc is None:
                    continue
                candidates[doc_id] = {**doc, "scores": {}}
            candidates[doc_id]["scores"]["semantic"] = similarity

        # 3. Graph boost (connection count)
        for doc_id, cand in candidates.items():
            rels = self.store.get_relationships(doc_id)
            connection_count = len(rels)
            # Logarithmic boost: 0 connections = 0, 10 = ~0.7, 100 = ~1.0
            cand["scores"]["graph"] = min(math.log1p(connection_count) / 5.0, 1.0)

        # 4. Recency boost
        self._apply_recency_scores(candidates)

        # Apply filters
        if entity:
            candidates = self._filter_by_entity(candidates, entity)
        if doc_type:
            candidates = {k: v for k, v in candidates.items() if v.get("type") == doc_type}
        if project:
            candidates = {k: v for k, v in candidates.items() if v.get("project") == project}

        # Compute combined scores
        results = []
        for _doc_id, cand in candidates.items():
            scores = cand.get("scores", {})
            combined = sum(
                scores.get(signal, 0.0) * self.weights.get(signal, 0.0) for signal in self.weights
            )
            cand["score"] = round(combined, 4)
            cand["score_breakdown"] = {k: round(v, 4) for k, v in scores.items()}
            results.append(cand)

        # Sort by combined score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Trim to limit
        results = results[:limit]

        # Clean up internal scoring data
        for r in results:
            r.pop("scores", None)

        duration_ms = (time.monotonic() - start) * 1000

        # Log query
        self.store.content.log_query(
            tool="hybrid_search",
            params={"query": query, "doc_type": doc_type, "project": project},
            result_ids=[r["id"] for r in results],
            duration_ms=duration_ms,
        )

        logger.debug(
            "Hybrid search '%s': %d results in %.1fms",
            query,
            len(results),
            duration_ms,
        )
        return results

    def _semantic_search(self, query: str, limit: int = 60) -> list[tuple[str, float]]:
        """Find documents by embedding similarity.

        Returns:
            List of (doc_id, similarity) tuples, sorted by similarity descending.
        """
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return []

        # Get all embeddings and compute cosine similarity
        rows = self.store.content.get_all_embeddings(limit=10000)

        scored: list[tuple[str, float]] = []
        for row in rows:
            try:
                doc_embedding = struct.unpack(f"{row['dimensions']}f", row["embedding"])
            except struct.error:
                logger.warning("Corrupted embedding for doc_id=%s, skipping", row["doc_id"])
                continue
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            scored.append((row["doc_id"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _embed_query(self, query: str) -> tuple[float, ...] | None:
        """Generate embedding for a query string."""
        if self._embedding_provider is None:
            return None
        try:
            vector = self._embedding_provider.embed(query)
            if vector is None:
                return None
            return tuple(vector)
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _apply_recency_scores(self, candidates: dict[str, dict[str, Any]]) -> None:
        """Apply recency boost based on created_at timestamp."""
        import datetime

        now = datetime.datetime.now(datetime.UTC)
        for cand in candidates.values():
            created = cand.get("created_at", "")
            if not created:
                cand.setdefault("scores", {})["recency"] = 0.0
                continue
            try:
                dt = datetime.datetime.fromisoformat(created)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.UTC)
                age_days = (now - dt).total_seconds() / 86400
                # Exponential decay: 0 days = 1.0, 7 days ≈ 0.8, 30 days ≈ 0.4
                score = math.exp(-0.03 * age_days)
                cand.setdefault("scores", {})["recency"] = score
            except (ValueError, TypeError):
                cand.setdefault("scores", {})["recency"] = 0.0

    def _filter_by_entity(
        self, candidates: dict[str, dict[str, Any]], entity_name: str
    ) -> dict[str, dict[str, Any]]:
        """Filter candidates to only those mentioning a specific entity."""
        # Find entity by name
        entities = self.store.graph.list_entities()
        entity_ids = [e["id"] for e in entities if e["name"].lower() == entity_name.lower()]
        if not entity_ids:
            return candidates

        # Get all objects mentioning this entity
        mentioning_ids: set[str] = set()
        for eid in entity_ids:
            mentioning_ids.update(self.store.graph.get_entity_mentions(eid))

        return {k: v for k, v in candidates.items() if k in mentioning_ids}
