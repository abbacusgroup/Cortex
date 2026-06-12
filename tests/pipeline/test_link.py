"""Tests for cortex.pipeline.link (LinkStage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.link import LinkStage
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
def linker(store: Store) -> LinkStage:
    """LinkStage with no LLM configured (fallback mode)."""
    llm = LLMClient(store.config)
    return LinkStage(store, llm)


def _create_obj(store: Store, *, title: str = "Test", obj_type: str = "idea") -> str:
    return store.create(obj_type=obj_type, title=title, content="Body")


# -- Entity resolution ------------------------------------------------------


class TestEntityResolution:
    def test_entities_created_in_graph(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store, title="Python guide")
        entities = [
            {"name": "Python", "type": "technology"},
            {"name": "CQRS", "type": "pattern"},
        ]

        result = linker.run(obj_id, entities)
        assert result["entities_resolved"] == 2

        # Verify entities exist in graph
        all_entities = store.list_entities()
        names = [e["name"] for e in all_entities]
        assert "Python" in names
        assert "CQRS" in names

    def test_mentions_added_for_entities(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store, title="Mentions test")
        entities = [{"name": "Docker", "type": "technology"}]

        result = linker.run(obj_id, entities)
        entity_id = result["entities"][0]["entity_id"]

        mentions = store.graph.get_entity_mentions(entity_id)
        assert obj_id in mentions

    def test_duplicate_entity_names_deduped(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store, title="Dedup test")
        entities = [
            {"name": "React", "type": "technology"},
            {"name": "React", "type": "technology"},
        ]

        result = linker.run(obj_id, entities)
        # Both resolve to the same entity_id
        ids = [e["entity_id"] for e in result["entities"]]
        assert ids[0] == ids[1]

    def test_entity_from_previous_run_is_reused(self, store: Store, linker: LinkStage):
        obj_a = _create_obj(store, title="First")
        obj_b = _create_obj(store, title="Second")

        linker.run(obj_a, [{"name": "Kubernetes", "type": "technology"}])
        result = linker.run(obj_b, [{"name": "Kubernetes", "type": "technology"}])

        # Same entity reused — only one entity with that name
        k8s_entities = [e for e in store.list_entities() if e["name"] == "Kubernetes"]
        assert len(k8s_entities) == 1

        # Second object also linked
        entity_id = result["entities"][0]["entity_id"]
        mentions = store.graph.get_entity_mentions(entity_id)
        assert obj_b in mentions


# -- Empty entities ---------------------------------------------------------


class TestEmptyEntities:
    def test_no_entities_created_with_empty_list(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store)
        result = linker.run(obj_id, [])
        assert result["entities_resolved"] == 0
        assert result["entities"] == []

    def test_entities_with_blank_name_skipped(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store)
        entities = [{"name": "", "type": "technology"}]
        result = linker.run(obj_id, entities)
        assert result["entities_resolved"] == 0


# -- Entity type defaulting -------------------------------------------------


class TestEntityTypeDefault:
    def test_unknown_entity_type_defaults_to_concept(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store, title="Type default")
        entities = [{"name": "Something", "type": "alien"}]

        result = linker.run(obj_id, entities)
        # GraphStore.create_entity normalizes unknown types to "concept"
        assert result["entities_resolved"] == 1

    def test_missing_type_defaults_to_concept(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store, title="No type field")
        entities = [{"name": "Widget"}]

        result = linker.run(obj_id, entities)
        assert result["entities_resolved"] == 1
        assert result["entities"][0]["type"] == "concept"


# -- Relationship discovery (no LLM) ---------------------------------------


class TestNoLlmRelationships:
    def test_no_relationships_without_llm(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store, title="Lonely object")
        result = linker.run(obj_id, [])
        assert result["relationships_created"] == 0
        assert result["relationships"] == []


# -- Pipeline stage ---------------------------------------------------------


class TestPipelineStage:
    def test_pipeline_stage_updated_to_linked(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store, title="Stage test")
        linker.run(obj_id, [])

        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["pipeline_stage"] == "linked"

    def test_result_status_is_linked(self, store: Store, linker: LinkStage):
        obj_id = _create_obj(store)
        result = linker.run(obj_id, [])
        assert result["status"] == "linked"


# -- Relationship discovery (with a fake LLM) -------------------------------


class _FakeLLM:
    """Stand-in LLMClient returning a scripted relationship list."""

    available = True

    def __init__(self, rels):
        self._rels = rels
        self.captured: dict = {}

    def discover_relationships(self, *, new_id, new_title, new_type, new_content, existing):
        self.captured = {"new_id": new_id, "existing": existing}
        return self._rels


class TestRelationshipDiscovery:
    def test_edge_to_new_object_survives_and_persists_provenance(self, store: Store):
        """An LLM edge that references the new object's real ID must be created
        (regression: it used to be dropped), with confidence + provenance."""
        new_id = _create_obj(store, title="New thing")
        other_id = _create_obj(store, title="Existing thing")

        rels = [
            {
                "from_id": new_id,
                "to_id": other_id,
                "rel_type": "supports",
                "confidence": 0.82,
            }
        ]
        linker = LinkStage(store, _FakeLLM(rels))
        result = linker.run(new_id, [])

        assert result["relationships_created"] == 1

        graph_rels = store.get_relationships(new_id)
        assert any(
            r["other_id"] == other_id and r["rel_type"] == "supports"
            for r in graph_rels
        )

        prov = store.graph.get_relationship_provenance(
            from_id=new_id, rel_type="supports", to_id=other_id
        )
        assert prov is not None
        assert prov["inferred_by"] == "llm"
        assert abs(prov["confidence"] - 0.82) < 1e-6

    def test_edge_between_two_existing_objects_is_rejected(self, store: Store):
        """An LLM edge touching neither endpoint of the new object must be
        dropped — the LLM must not silently rewire unrelated documents."""
        new_id = _create_obj(store, title="New thing")
        a_id = _create_obj(store, title="A")
        b_id = _create_obj(store, title="B")

        rels = [
            {
                "from_id": a_id,
                "to_id": b_id,
                "rel_type": "supports",
                "confidence": 0.9,
            }
        ]
        linker = LinkStage(store, _FakeLLM(rels))
        result = linker.run(new_id, [])

        assert result["relationships_created"] == 0
        # The bogus edge was never written.
        assert store.get_relationships(a_id) == []

    def test_edge_to_nonexistent_object_is_rejected(self, store: Store):
        new_id = _create_obj(store, title="New thing")
        rels = [
            {
                "from_id": new_id,
                "to_id": "does-not-exist",
                "rel_type": "supports",
                "confidence": 0.9,
            }
        ]
        linker = LinkStage(store, _FakeLLM(rels))
        result = linker.run(new_id, [])
        assert result["relationships_created"] == 0
