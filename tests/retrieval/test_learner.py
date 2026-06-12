"""Tests for cortex.retrieval.learner (LearningLoop)."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.retrieval.engine import DEFAULT_WEIGHTS, RetrievalEngine
from cortex.retrieval.learner import (
    LAST_ACCESS_PREFIX,
    MIN_SIGNAL_WEIGHT,
    LearningLoop,
)

ONTOLOGY_PATH = find_ontology()


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Initialized Store backed by tmp_path."""
    cfg = CortexConfig(data_dir=tmp_path)
    s = Store(cfg)
    s.initialize(ONTOLOGY_PATH)
    return s


@pytest.fixture()
def loop(store: Store) -> LearningLoop:
    return LearningLoop(store)


def _obj(
    store: Store,
    title: str = "Obj",
    obj_type: str = "decision",
    project: str = "test",
    tier: str = "archive",
) -> str:
    return store.create(
        obj_type=obj_type,
        title=title,
        content=title,
        project=project,
        tier=tier,
    )


# ── Access Recording ────────────────────────────────────────────────


class TestAccessRecording:
    def test_record_access_increments_count(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        obj_id = _obj(store)
        loop.record_access(obj_id)
        assert loop.get_access_count(obj_id) == 1

    def test_multiple_accesses_increment(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        obj_id = _obj(store)
        for _ in range(5):
            loop.record_access(obj_id)
        assert loop.get_access_count(obj_id) == 5

    def test_ten_accesses_promotes_to_reflex(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """10+ accesses triggers promotion to reflex tier."""
        obj_id = _obj(store, tier="archive")
        for _ in range(10):
            loop.record_access(obj_id)

        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["tier"] == "reflex"

    def test_get_access_count_unaccessed_is_zero(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        obj_id = _obj(store)
        assert loop.get_access_count(obj_id) == 0


# ── Miss Detection ──────────────────────────────────────────────────


class TestMissDetection:
    def test_miss_when_read_not_in_results(
        self,
        loop: LearningLoop,
    ):
        """Read ID not in search results -> miss."""
        assert (
            loop.detect_miss(
                context_query="some query",
                context_result_ids=["id-1", "id-2"],
                subsequent_read_id="id-99",
            )
            is True
        )

    def test_hit_when_read_in_results(
        self,
        loop: LearningLoop,
    ):
        """Read ID present in search results -> not a miss."""
        assert (
            loop.detect_miss(
                context_query="some query",
                context_result_ids=["id-1", "id-2", "id-3"],
                subsequent_read_id="id-2",
            )
            is False
        )


# ── Tier Adjustment ─────────────────────────────────────────────────


class TestTierAdjustment:
    def test_promote_high_access_objects(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """Object with 10+ accesses gets promoted via adjust_tiers."""
        obj_id = _obj(store, tier="archive")
        for _ in range(10):
            loop.record_access(obj_id)

        # Reset tier to archive to test adjust_tiers path
        store.content.update(obj_id, tier="archive")

        result = loop.adjust_tiers()
        assert result["promoted"] >= 1

        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["tier"] == "reflex"

    def test_demote_inactive_reflex(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """Reflex object with old last_access gets demoted."""
        obj_id = _obj(store, tier="reflex")

        # Manually set an old last_access timestamp (60 days ago)
        old_ts = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=60)).isoformat()
        store.content.set_config(
            f"{LAST_ACCESS_PREFIX}{obj_id}",
            old_ts,
        )

        result = loop.adjust_tiers(inactivity_days=30)
        assert result["demoted"] >= 1

        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["tier"] == "recall"

    def test_no_changes_when_all_current(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """No promotions or demotions when nothing qualifies."""
        _obj(store, tier="archive")
        result = loop.adjust_tiers()
        assert result == {"promoted": 0, "demoted": 0}


# ── Weights ─────────────────────────────────────────────────────────


class TestWeights:
    def test_get_weights_returns_defaults(
        self,
        loop: LearningLoop,
    ):
        weights = loop.get_weights()
        assert weights == dict(DEFAULT_WEIGHTS)

    def test_update_weights_persists(
        self,
        loop: LearningLoop,
    ):
        custom = {"keyword": 0.5, "semantic": 0.2, "graph": 0.2, "recency": 0.1}
        loop.update_weights(custom)
        assert loop.get_weights() == custom

    def test_reset_weights_restores_defaults(
        self,
        loop: LearningLoop,
    ):
        custom = {"keyword": 0.9, "semantic": 0.05, "graph": 0.03, "recency": 0.02}
        loop.update_weights(custom)

        restored = loop.reset_weights()
        assert restored == dict(DEFAULT_WEIGHTS)
        assert loop.get_weights() == dict(DEFAULT_WEIGHTS)

    def test_weights_persisted_in_config_store(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """Weights survive a new LearningLoop instance."""
        custom = {"keyword": 0.6, "semantic": 0.1, "graph": 0.2, "recency": 0.1}
        loop.update_weights(custom)

        loop2 = LearningLoop(store)
        assert loop2.get_weights() == custom

    def test_corrupt_persisted_weights_return_defaults(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """get_weights validates the stored JSON instead of returning junk."""
        from cortex.retrieval.learner import WEIGHTS_CONFIG_KEY

        store.content.set_config(WEIGHTS_CONFIG_KEY, "not json at all")
        assert loop.get_weights() == dict(DEFAULT_WEIGHTS)


# ── Miss-driven weight adaptation ───────────────────────────────────


class TestRecordMiss:
    def test_hit_does_not_change_weights(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        obj_id = _obj(store, title="Alpha topic")
        adapted = loop.record_miss(
            context_query="alpha",
            context_result_ids=[obj_id],
            subsequent_read_id=obj_id,
        )
        assert adapted is False
        assert loop.get_weights() == dict(DEFAULT_WEIGHTS)

    def test_miss_on_lexical_match_boosts_keyword(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """Keyword search finds the doc, but results didn't -> boost keyword."""
        obj_id = _obj(store, title="Zebra migration patterns")
        adapted = loop.record_miss(
            context_query="zebra",
            context_result_ids=["some-other-id"],
            subsequent_read_id=obj_id,
        )
        assert adapted is True
        weights = loop.get_weights()
        assert weights["keyword"] > DEFAULT_WEIGHTS["keyword"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_miss_on_lexical_mismatch_boosts_semantic(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """Doc shares no vocabulary with the query -> boost semantic."""
        obj_id = _obj(store, title="Quarterly budget review")
        adapted = loop.record_miss(
            context_query="finances spending plan",
            context_result_ids=["some-other-id"],
            subsequent_read_id=obj_id,
        )
        assert adapted is True
        weights = loop.get_weights()
        assert weights["semantic"] > DEFAULT_WEIGHTS["semantic"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_weights_stay_bounded_after_many_misses(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        obj_id = _obj(store, title="Walrus habitat notes")
        for _ in range(50):
            loop.record_miss(
                context_query="walrus",
                context_result_ids=["nope"],
                subsequent_read_id=obj_id,
            )
        weights = loop.get_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-2)
        for signal, value in weights.items():
            # Post-normalization floor: MIN_SIGNAL_WEIGHT shaved by at most
            # the boost overshoot (total <= 1 + learning_rate).
            assert value >= MIN_SIGNAL_WEIGHT * 0.9, signal
            assert value < 1.0, signal
        # The boosted signal actually grew
        assert weights["keyword"] > DEFAULT_WEIGHTS["keyword"]

    def test_engine_picks_up_miss_adapted_weights(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """End-to-end: a recorded miss changes the engine's effective weights."""
        obj_id = _obj(store, title="Glacier melt observations")
        loop.record_miss(
            context_query="glacier",
            context_result_ids=["nope"],
            subsequent_read_id=obj_id,
        )
        engine = RetrievalEngine(store)
        engine.search("glacier")
        assert engine.weights == loop.get_weights()
        assert engine.weights != dict(DEFAULT_WEIGHTS)


# ── Tier writes are dual-written (graph stays in sync) ──────────────


class TestTierDualWrite:
    def test_promotion_updates_graph_tier(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        """record_access promotion goes through Store.update -> graph tier."""
        obj_id = _obj(store, tier="archive")
        for _ in range(10):
            loop.record_access(obj_id)

        assert store.content.get(obj_id)["tier"] == "reflex"
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj.get("tier") == "reflex"

    def test_adjust_tiers_promotion_updates_graph(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        obj_id = _obj(store, tier="archive")
        for _ in range(10):
            loop.record_access(obj_id)
        store.update(obj_id, tier="archive")  # reset both stores

        result = loop.adjust_tiers()
        assert result["promoted"] >= 1
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj.get("tier") == "reflex"

    def test_adjust_tiers_demotion_updates_graph(
        self,
        store: Store,
        loop: LearningLoop,
    ):
        obj_id = _obj(store, tier="reflex")
        old_ts = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=60)).isoformat()
        store.content.set_config(f"{LAST_ACCESS_PREFIX}{obj_id}", old_ts)

        result = loop.adjust_tiers(inactivity_days=30)
        assert result["demoted"] >= 1
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj.get("tier") == "recall"
