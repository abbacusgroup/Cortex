"""Learning loop — implicit feedback, tier adjustment, ranking adaptation.

Learns from usage patterns to improve retrieval over time:
- Detects miss signals (context → search → read patterns)
- Adjusts tiers based on access frequency
- Tunes ranking weights based on feedback
"""

from __future__ import annotations

import json

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.retrieval.engine import DEFAULT_WEIGHTS

logger = get_logger("retrieval.learner")

# Config keys for persisted weights
WEIGHTS_CONFIG_KEY = "retrieval_weights"
ACCESS_COUNT_PREFIX = "access_count:"
LAST_ACCESS_PREFIX = "last_access:"


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

            # Promote: high access count
            if tier != "reflex" and access_count >= 10:
                try:
                    self.store.content.update(obj_id, tier="reflex")
                    promoted += 1
                except Exception:
                    pass

            # Demote: reflex but not accessed recently
            elif tier == "reflex" and last_access and last_access < cutoff_str:
                try:
                    self.store.content.update(obj_id, tier="recall")
                    demoted += 1
                except Exception:
                    pass

        return {"promoted": promoted, "demoted": demoted}

    def get_weights(self) -> dict[str, float]:
        """Get current ranking weights (persisted or defaults)."""
        raw = self.store.content.get_config(WEIGHTS_CONFIG_KEY, "")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return dict(DEFAULT_WEIGHTS)

    def update_weights(self, weights: dict[str, float]) -> None:
        """Persist updated ranking weights."""
        self.store.content.set_config(WEIGHTS_CONFIG_KEY, json.dumps(weights))

    def reset_weights(self) -> dict[str, float]:
        """Reset ranking weights to defaults."""
        defaults = dict(DEFAULT_WEIGHTS)
        self.update_weights(defaults)
        return defaults

    def _maybe_promote(self, obj_id: str) -> None:
        """Promote an object to reflex tier if it meets criteria."""
        doc = self.store.content.get(obj_id)
        if doc and doc.get("tier") != "reflex":
            try:
                self.store.content.update(obj_id, tier="reflex")
                logger.info("Promoted %s to reflex tier", obj_id)
            except Exception:
                pass
