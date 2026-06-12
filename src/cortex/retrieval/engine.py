"""Hybrid retrieval engine — keyword + semantic + graph-aware ranking.

Combines FTS5 (BM25), embedding cosine similarity, and graph connectivity
into a single ranked result set with configurable weights.

Unless explicit ``weights`` are passed, the engine reads the adaptive
weights persisted by :class:`cortex.retrieval.learner.LearningLoop` on every
search, falling back to ``DEFAULT_WEIGHTS`` until feedback has moved them.
"""

from __future__ import annotations

import json
import math
import struct
import time
from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.services.embeddings import (
    EmbeddingProvider,
    check_embedding_model_consistency,
)

logger = get_logger("retrieval.engine")

# Default ranking weights
DEFAULT_WEIGHTS = {
    "keyword": 0.4,
    "semantic": 0.3,
    "graph": 0.2,
    "recency": 0.1,
}

# Config key under which LearningLoop persists adapted ranking weights.
# Defined here (not in learner.py) so the engine can read it without a
# circular import; learner.py re-exports it for backwards compatibility.
WEIGHTS_CONFIG_KEY = "retrieval_weights"

# Minimum combined score a result must reach to be returned. 0.0 = off
# (every candidate passes). Kept off by default so ranking output only
# changes when a caller opts in; see the `min_relevance` parameters.
DEFAULT_MIN_RELEVANCE = 0.0


def load_persisted_weights(content_store: Any) -> dict[str, float]:
    """Load the learner-persisted ranking weights, validated.

    Returns a complete weight dict (every signal in ``DEFAULT_WEIGHTS``
    present). Falls back to ``DEFAULT_WEIGHTS`` when nothing is persisted,
    the stored JSON is corrupt, the shape is wrong, any value is not a
    non-negative number, or all weights are zero — logging a warning for
    every case that isn't simply "nothing persisted yet".
    """
    raw = content_store.get_config(WEIGHTS_CONFIG_KEY, "")
    if not raw:
        return dict(DEFAULT_WEIGHTS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(
            "Persisted retrieval weights are corrupt JSON (%s) — using defaults", e
        )
        return dict(DEFAULT_WEIGHTS)
    if not isinstance(parsed, dict):
        logger.warning(
            "Persisted retrieval weights have wrong shape (%s) — using defaults",
            type(parsed).__name__,
        )
        return dict(DEFAULT_WEIGHTS)

    weights = dict(DEFAULT_WEIGHTS)
    for signal in weights:
        if signal not in parsed:
            continue
        value = parsed[signal]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            logger.warning(
                "Persisted retrieval weight %r has invalid value %r — using defaults",
                signal,
                value,
            )
            return dict(DEFAULT_WEIGHTS)
        weights[signal] = float(value)

    if sum(weights.values()) <= 0:
        logger.warning("Persisted retrieval weights are all zero — using defaults")
        return dict(DEFAULT_WEIGHTS)
    return weights


class RetrievalEngine:
    """Hybrid search combining keyword, semantic, graph, and recency signals."""

    def __init__(
        self,
        store: Store,
        weights: dict[str, float] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
    ):
        self.store = store
        # Explicit weights pin the engine; otherwise the learner-persisted
        # weights are re-resolved on every search (defaults until feedback
        # has moved them).
        self._explicit_weights = dict(weights) if weights else None
        self.weights = self._explicit_weights or dict(DEFAULT_WEIGHTS)
        self._embedding_provider = embedding_provider
        self.min_relevance = min_relevance
        self._model_consistency_checked = False

    def _resolve_weights(self) -> dict[str, float]:
        """Resolve the ranking weights for this search.

        Explicit constructor weights win; otherwise the weights persisted
        by the LearningLoop are used (defaults when none exist).
        """
        if self._explicit_weights is not None:
            return dict(self._explicit_weights)
        return load_persisted_weights(self.store.content)

    def search(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        project: str | None = None,
        entity: str | None = None,
        limit: int = 20,
        min_relevance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a hybrid search.

        Args:
            query: Search query text.
            doc_type: Filter by knowledge type.
            project: Filter by project.
            entity: Filter by entity name.
            limit: Maximum results to return.
            min_relevance: Per-call minimum combined score; results below
                it are dropped. None uses the engine default (0.0 = off).

        Returns:
            List of result dicts sorted by combined score (best first).
        """
        start = time.monotonic()

        if not query or not query.strip():
            return []

        weights = self._resolve_weights()
        # Keep the public attribute in sync for introspection/debugging.
        self.weights = weights
        relevance_floor = self.min_relevance if min_relevance is None else min_relevance

        # Gather candidates from multiple signals
        candidates: dict[str, dict[str, Any]] = {}

        # 1. Keyword search (FTS5, BM25-ranked)
        keyword_results = self.store.search(
            query, doc_type=doc_type, project=project, limit=limit * 3
        )
        keyword_scores = self._keyword_scores(keyword_results)
        for doc in keyword_results:
            doc_id = doc["id"]
            if doc_id not in candidates:
                candidates[doc_id] = {**doc, "scores": {}}
            candidates[doc_id]["scores"]["keyword"] = keyword_scores[doc_id]

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
            # Logarithmic boost: 0 connections = 0, 10 ≈ 0.48, 100 ≈ 0.92 (capped at 1.0)
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
                scores.get(signal, 0.0) * weights.get(signal, 0.0) for signal in weights
            )
            cand["score"] = round(combined, 4)
            cand["score_breakdown"] = {k: round(v, 4) for k, v in scores.items()}
            results.append(cand)

        # Sort by combined score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Drop candidates below the relevance floor (before trimming), so
        # briefings aren't padded with unrelated docs just to fill `limit`.
        if relevance_floor > 0:
            results = [r for r in results if r["score"] >= relevance_floor]

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

    @staticmethod
    def _keyword_scores(keyword_results: list[dict[str, Any]]) -> dict[str, float]:
        """Normalize FTS5 BM25 values into [0, 1] keyword scores.

        SQLite's ``bm25()`` returns *negative* values where more negative is
        a better match; ``store.search`` surfaces it as ``rank``. Scores are
        normalized against the best match in the result set, so the keyword
        score reflects actual BM25 magnitude instead of mere list position.
        Falls back to linear rank-position normalization when no usable
        BM25 values are present (e.g. a store stub without ``rank``).
        """
        bm25_raw: dict[str, float] = {}
        for doc in keyword_results:
            rank = doc.get("rank")
            if isinstance(rank, (int, float)) and not isinstance(rank, bool):
                bm25_raw[doc["id"]] = max(-float(rank), 0.0)
        max_raw = max(bm25_raw.values(), default=0.0)

        scores: dict[str, float] = {}
        for i, doc in enumerate(keyword_results):
            doc_id = doc["id"]
            if max_raw > 0 and doc_id in bm25_raw:
                scores[doc_id] = min(bm25_raw[doc_id] / max_raw, 1.0)
            else:
                # Positional fallback: first result = 1.0, linearly decreasing
                scores[doc_id] = 1.0 - (i / max(len(keyword_results), 1))
        return scores

    def _semantic_search(self, query: str, limit: int = 60) -> list[tuple[str, float]]:
        """Find documents by embedding similarity.

        Returns:
            List of (doc_id, similarity) tuples, sorted by similarity descending.
        """
        self._warn_on_model_mismatch_once()
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
        except Exception as e:
            # A silently-disabled semantic path is indistinguishable from
            # "no semantic matches" — log so degradation is visible.
            logger.warning(
                "Query embedding failed (model=%s) — semantic ranking disabled "
                "for this query: %s",
                getattr(self._embedding_provider, "model_name", "unknown"),
                e,
            )
            return None

    def _warn_on_model_mismatch_once(self) -> None:
        """Warn (once per engine) if stored embeddings don't match the model.

        Mismatched models make cosine similarity meaningless; surfacing it
        here means every deployment path that runs a semantic search gets
        the warning, regardless of how the engine was constructed.
        """
        if self._model_consistency_checked or self._embedding_provider is None:
            return
        self._model_consistency_checked = True
        try:
            warning = check_embedding_model_consistency(
                self.store.content, self._embedding_provider
            )
        except Exception as e:
            logger.debug("Embedding model consistency check failed: %s", e)
            return
        if warning:
            logger.warning("%s", warning)

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
