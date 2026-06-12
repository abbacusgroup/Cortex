"""Learning loop — implicit feedback, tier adjustment, ranking adaptation.

Learns from usage patterns to improve retrieval over time:
- Detects miss signals (context → search → read patterns)
- Adjusts tiers based on access frequency
- Tunes ranking weights based on feedback

The weights persisted here are read back by ``RetrievalEngine`` on every
search (unless the engine was constructed with explicit weights), so
``record_miss``/``update_weights`` directly shape ranking.
"""

from __future__ import annotations

import json

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.retrieval.engine import (
    DEFAULT_WEIGHTS,
    WEIGHTS_CONFIG_KEY,
    load_persisted_weights,
)

logger = get_logger("retrieval.learner")

__all__ = [
    "ACCESS_COUNT_PREFIX",
    "LAST_ACCESS_PREFIX",
    "WEIGHTS_CONFIG_KEY",
    "LearningLoop",
]

# Config keys for access tracking (weights key lives in engine.py)
ACCESS_COUNT_PREFIX = "access_count:"
LAST_ACCESS_PREFIX = "last_access:"

# Bounds for adaptive weight nudging: no signal may be silenced entirely
# (floor) and a single miss only moves weights a little (learning rate).
MIN_SIGNAL_WEIGHT = 0.05
DEFAULT_LEARNING_RATE = 0.05


class LearningLoop:
    """Learns from query patterns to improve retrieval."""

    def __init__(self, store: Store):
        self.store = store

    def record_access(self, obj_id: str) -> None:
        """Record that an object was accessed (read/viewed)."""
        count_key = f"{ACCESS_COUNT_PREFIX}{obj_id}"
        current = int(self.store.content.get_config(count_key, "0"))
        self.store.content.set_config(count_key, str(current + 1))

        import datetime

        now = datetime.datetime.now(datetime.UTC).isoformat()
        self.store.content.set_config(f"{LAST_ACCESS_PREFIX}{obj_id}", now)

        # Check for promotion threshold
        if current + 1 >= 10:
            self._maybe_promote(obj_id)

    def get_access_count(self, obj_id: str) -> int:
        """Get total access count for an object."""
        count_key = f"{ACCESS_COUNT_PREFIX}{obj_id}"
        return int(self.store.content.get_config(count_key, "0"))

    def detect_miss(
        self,
        *,
        context_query: str,
        context_result_ids: list[str],
        subsequent_read_id: str,
    ) -> bool:
        """Detect a miss signal: user searched, then read an object not in results.

        A miss means the retrieval didn't surface the right result, and the
        user had to search again or navigate directly.

        Returns:
            True if this is a miss signal.
        """
        return subsequent_read_id not in context_result_ids

    def record_miss(
        self,
        *,
        context_query: str,
        context_result_ids: list[str],
        subsequent_read_id: str,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> bool:
        """Record a retrieval miss and adapt the ranking weights.

        This is the feedback path that actually drives ``update_weights``:
        when the object the user ended up reading was missing from the
        results, the miss is diagnosed and the corresponding signal weight
        gets a small, bounded boost (weights are re-normalized to sum 1.0,
        with a floor so no signal is ever silenced):

        - If a plain keyword search *does* find the object, the hybrid blend
          buried a lexical match → boost ``keyword``.
        - If keyword search can't find it either, the query and document
          don't share vocabulary → boost ``semantic``.

        Returns:
            True if a miss was detected (and weights adapted), False if the
            read object was in the results (no adaptation).
        """
        if not self.detect_miss(
            context_query=context_query,
            context_result_ids=context_result_ids,
            subsequent_read_id=subsequent_read_id,
        ):
            return False

        try:
            keyword_ids = {d["id"] for d in self.store.search(context_query, limit=50)}
        except Exception as e:
            logger.warning(
                "Miss recorded for %r but keyword diagnosis failed — "
                "weights left unchanged: %s",
                context_query,
                e,
            )
            return True

        boosted = "keyword" if subsequent_read_id in keyword_ids else "semantic"
        weights = self.get_weights()
        weights[boosted] = weights.get(boosted, 0.0) + learning_rate
        weights = self._rebalance(weights)
        self.update_weights(weights)
        logger.info(
            "Retrieval miss for %r (read %s) — boosted %s weight to %.3f",
            context_query,
            subsequent_read_id,
            boosted,
            weights[boosted],
        )
        return True

    @staticmethod
    def _rebalance(weights: dict[str, float]) -> dict[str, float]:
        """Clamp each signal to the floor, then normalize the sum to 1.0.

        Normalization can shave a clamped value slightly below
        ``MIN_SIGNAL_WEIGHT`` (worst case ``MIN_SIGNAL_WEIGHT / total``),
        but every signal always keeps a strictly positive share — no signal
        can ever be silenced by feedback.
        """
        clamped = {
            signal: max(weights.get(signal, 0.0), MIN_SIGNAL_WEIGHT)
            for signal in DEFAULT_WEIGHTS
        }
        total = sum(clamped.values())
        return {signal: round(value / total, 4) for signal, value in clamped.items()}

    def adjust_tiers(self, *, inactivity_days: int = 30) -> dict[str, int]:
        """Adjust tiers based on access patterns.

        - Promote frequently accessed objects to reflex
        - Demote inactive reflex objects to recall

        Returns:
            Dict with promoted and demoted counts.
        """
        import datetime

        promoted = 0
        demoted = 0
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=inactivity_days)
        cutoff_str = cutoff.isoformat()

        all_objects = self.store.list_objects(limit=1000)
        for obj in all_objects:
            obj_id = obj.get("id", "")
            tier = obj.get("tier", "archive")
            access_count = self.get_access_count(obj_id)
            last_access = self.store.content.get_config(f"{LAST_ACCESS_PREFIX}{obj_id}", "")

            # Promote: high access count. Tier writes go through Store.update
            # (the single dual-write path) so the graph tier stays in sync.
            if tier != "reflex" and access_count >= 10:
                try:
                    self.store.update(obj_id, tier="reflex")
                    promoted += 1
                except Exception as e:
                    logger.warning("Failed to promote %s to reflex: %s", obj_id, e)

            # Demote: reflex but not accessed recently
            elif tier == "reflex" and last_access and last_access < cutoff_str:
                try:
                    self.store.update(obj_id, tier="recall")
                    demoted += 1
                except Exception as e:
                    logger.warning("Failed to demote %s from reflex: %s", obj_id, e)

        return {"promoted": promoted, "demoted": demoted}

    def get_weights(self) -> dict[str, float]:
        """Get current ranking weights (persisted or defaults).

        Uses the same validated loader as ``RetrievalEngine``, so the
        learner and the engine always agree on the effective weights.
        """
        return load_persisted_weights(self.store.content)

    def update_weights(self, weights: dict[str, float]) -> None:
        """Persist updated ranking weights."""
        self.store.content.set_config(WEIGHTS_CONFIG_KEY, json.dumps(weights))

    def reset_weights(self) -> dict[str, float]:
        """Reset ranking weights to defaults."""
        defaults = dict(DEFAULT_WEIGHTS)
        self.update_weights(defaults)
        return defaults

    def _maybe_promote(self, obj_id: str) -> None:
        """Promote an object to reflex tier if it meets criteria.

        Goes through Store.update (the single dual-write path) so the
        graph's tier predicate is updated alongside SQLite.
        """
        doc = self.store.content.get(obj_id)
        if doc and doc.get("tier") != "reflex":
            try:
                self.store.update(obj_id, tier="reflex")
                logger.info("Promoted %s to reflex tier", obj_id)
            except Exception as e:
                logger.warning("Failed to promote %s to reflex tier: %s", obj_id, e)
