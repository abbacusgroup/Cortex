"""Tests for the EnrichStage pipeline stage."""

from __future__ import annotations

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.enrich import EnrichStage


@pytest.fixture
def store(tmp_path):
    config = CortexConfig(data_dir=tmp_path)
    s = Store(config)
    s.initialize(find_ontology())
    return s


@pytest.fixture
def enricher(store):
    return EnrichStage(store)


def _create_object(store, **overrides):
    """Helper to create a knowledge object with sensible defaults."""
    defaults = {
        "obj_type": "idea",
        "title": "Test object",
        "content": "Some content",
    }
    defaults.update(overrides)
    return store.create(**defaults)


class TestEnrichTierComputation:
    """Tests for _compute_tier logic."""

    def test_no_summary_gives_archive(self, store, enricher):
        """Object without summary or with confidence=0 -> archive."""
        obj_id = _create_object(store, confidence=0.0)
        result = enricher.run(obj_id)

        assert result["status"] == "enriched"
        assert result["tier"] == "archive"

    def test_classified_object_gives_recall(self, store, enricher):
        """Object with summary and confidence > 0 -> recall."""
        obj_id = _create_object(store)
        # Simulate classification: add summary + confidence
        store.content.update(
            obj_id, summary="A useful summary", confidence=0.8
        )
        result = enricher.run(obj_id)

        assert result["tier"] == "recall"

    def test_reflex_stays_reflex(self, store, enricher):
        """Reflex tier is never demoted by enrich (learning loop's job)."""
        obj_id = _create_object(store, tier="reflex")
        # Even without summary, reflex stays reflex
        store.content.update(obj_id, summary="", confidence=0.0)
        result = enricher.run(obj_id)

        assert result["tier"] == "reflex"

    def test_no_summary_with_high_confidence_still_archive(
        self, store, enricher
    ):
        """Empty summary with confidence > 0 is still archive."""
        obj_id = _create_object(store, confidence=0.9)
        store.content.update(obj_id, summary="")
        result = enricher.run(obj_id)

        assert result["tier"] == "archive"


class TestPromoteDemote:
    """Tests for explicit tier promotion/demotion."""

    def test_promote_to_reflex(self, store, enricher):
        obj_id = _create_object(store)
        assert enricher.promote_to_reflex(obj_id) is True

        doc = store.content.get(obj_id)
        assert doc["tier"] == "reflex"

    def test_demote_from_reflex(self, store, enricher):
        obj_id = _create_object(store, tier="reflex")
        assert enricher.demote_from_reflex(obj_id) is True

        doc = store.content.get(obj_id)
        assert doc["tier"] == "recall"


class TestConnectionCount:
    """Tests for connection counting."""

    def test_no_relationships_zero_connections(self, store, enricher):
        obj_id = _create_object(store)
        result = enricher.run(obj_id)

        assert result["connection_count"] == 0

    def test_connections_reflect_actual_relationships(
        self, store, enricher
    ):
        """connection_count matches the number of relationships."""
        obj_a = _create_object(store, title="Object A")
        obj_b = _create_object(store, title="Object B")
        obj_c = _create_object(store, title="Object C")

        store.create_relationship(
            from_id=obj_a, rel_type="supports", to_id=obj_b
        )
        store.create_relationship(
            from_id=obj_a, rel_type="causedBy", to_id=obj_c
        )

        result = enricher.run(obj_a)
        assert result["connection_count"] == 2


class TestStaleness:
    """Tests for staleness scoring."""

    def test_fresh_object_zero_staleness(self, store, enricher):
        obj_id = _create_object(store)
        result = enricher.run(obj_id)

        assert result["staleness_score"] == 0.0

    def test_stale_dependency_increases_staleness(self, store, enricher):
        """Object depending on a superseded object gets staleness bump."""
        dep_id = _create_object(store, title="Old dependency")
        obj_id = _create_object(store, title="Depends on old")

        store.create_relationship(
            from_id=obj_id, rel_type="dependsOn", to_id=dep_id
        )
        # Mark the dependency as superseded
        store.content.update(dep_id, pipeline_stage="superseded")

        result = enricher.run(obj_id)
        assert result["staleness_score"] > 0.0

    def test_staleness_capped_at_one(self, store, enricher):
        """Staleness score never exceeds 1.0."""
        obj_id = _create_object(store, title="Multi-dep")
        # Create 3 superseded dependencies (0.5 each -> 1.5, capped to 1.0)
        for i in range(3):
            dep_id = _create_object(store, title=f"Old dep {i}")
            store.create_relationship(
                from_id=obj_id, rel_type="dependsOn", to_id=dep_id
            )
            store.content.update(dep_id, pipeline_stage="superseded")

        result = enricher.run(obj_id)
        assert result["staleness_score"] == 1.0


class TestPipelineStage:
    """Tests for pipeline stage updates."""

    def test_enrichment_sets_pipeline_stage(self, store, enricher):
        """After enrichment, pipeline_stage is 'enriched'."""
        obj_id = _create_object(store)
        enricher.run(obj_id)

        doc = store.content.get(obj_id)
        assert doc["pipeline_stage"] == "enriched"

    def test_not_found_returns_not_found(self, store, enricher):
        """Enriching a nonexistent object returns not_found status."""
        result = enricher.run("nonexistent-id")
        assert result["status"] == "not_found"
