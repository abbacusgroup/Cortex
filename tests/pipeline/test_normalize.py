"""Tests for cortex.pipeline.normalize (NormalizeStage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.normalize import NormalizeStage
from cortex.services.llm import LLMClient

ONTOLOGY_PATH = find_ontology()


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Initialized Store backed by tmp_path."""
    cfg = CortexConfig(data_dir=tmp_path)
    s = Store(cfg)
    s.initialize(ONTOLOGY_PATH)
    return s


@pytest.fixture()
def normalizer(store: Store) -> NormalizeStage:
    """NormalizeStage with no LLM configured (fallback mode)."""
    llm = LLMClient(store.config)
    return NormalizeStage(store, llm)


def _create_obj(store: Store, *, title: str = "Test", content: str = "Body") -> str:
    return store.create(obj_type="idea", title=title, content=content)


# -- Basic normalization ----------------------------------------------------


class TestNormalize:
    def test_pipeline_stage_updated_to_normalized(self, store: Store, normalizer: NormalizeStage):
        obj_id = _create_obj(store, title="Stage test")
        normalizer.run(obj_id)

        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["pipeline_stage"] == "normalized"

    def test_returns_normalized_status(self, store: Store, normalizer: NormalizeStage):
        obj_id = _create_obj(store)
        result = normalizer.run(obj_id)
        assert result["status"] == "normalized"

    def test_returns_type_and_confidence(self, store: Store, normalizer: NormalizeStage):
        obj_id = _create_obj(store)
        result = normalizer.run(obj_id)
        assert "type" in result
        assert "confidence" in result


# -- Fallback mode (no LLM) ------------------------------------------------


class TestNormalizeFallback:
    def test_type_defaults_to_idea_without_llm(self, store: Store, normalizer: NormalizeStage):
        obj_id = _create_obj(store, title="Fallback test")
        result = normalizer.run(obj_id)
        assert result["type"] == "idea"
        assert result["confidence"] == 0.0

    def test_summary_set_after_normalization(self, store: Store, normalizer: NormalizeStage):
        obj_id = _create_obj(store, title="Summary check")
        normalizer.run(obj_id)

        doc = store.content.get(obj_id)
        assert doc is not None
        # In fallback mode, summary is set to the title
        assert doc["summary"] == "Summary check"

    def test_entities_empty_in_fallback(self, store: Store, normalizer: NormalizeStage):
        obj_id = _create_obj(store)
        result = normalizer.run(obj_id)
        assert result["entities"] == []

    def test_properties_empty_in_fallback(self, store: Store, normalizer: NormalizeStage):
        obj_id = _create_obj(store)
        result = normalizer.run(obj_id)
        assert result["properties"] == {}


# -- Non-existent object ----------------------------------------------------


class TestNormalizeNotFound:
    def test_nonexistent_object_returns_not_found(self, normalizer: NormalizeStage):
        result = normalizer.run("nonexistent-id-12345")
        assert result["status"] == "not_found"


# -- Pre-classified objects ---------------------------------------------------


class TestNormalizePreClassified:
    def test_pre_classified_skips_llm(self, store, normalizer):
        obj_id = store.create(
            obj_type="fix",
            title="Pre-classified",
            content="some content",
            summary="A pre-classified fix",
            confidence=0.9,
        )
        result = normalizer.run(obj_id)
        assert result["status"] == "normalized"
        assert result["type"] == "fix"
        assert result["confidence"] == 0.9

    def test_pre_classified_preserves_summary(self, store, normalizer):
        obj_id = store.create(
            obj_type="lesson",
            title="Test",
            content="content",
            summary="My custom summary",
        )
        normalizer.run(obj_id)
        doc = store.content.get(obj_id)
        assert doc["summary"] == "My custom summary"

    def test_without_summary_uses_fallback(self, store, normalizer):
        obj_id = store.create(obj_type="fix", title="No summary", content="content")
        result = normalizer.run(obj_id)
        assert result["status"] == "normalized"
        # Without LLM, fallback sets confidence to 0.0
        assert result["confidence"] == 0.0
