"""Tests for cortex.db.store (unified dual-write Store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.core.errors import NotFoundError
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology

ONTOLOGY_PATH = find_ontology()


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Initialized Store backed by tmp_path."""
    cfg = CortexConfig(data_dir=tmp_path)
    s = Store(cfg)
    s.initialize(ONTOLOGY_PATH)
    return s


def _create_sample(
    store: Store,
    *,
    title: str = "Sample",
    obj_type: str = "decision",
    content: str = "body",
    project: str = "cortex",
    tags: str = "test",
) -> str:
    return store.create(
        obj_type=obj_type,
        title=title,
        content=content,
        project=project,
        tags=tags,
    )


# ── Dual-write consistency ───────────────────────────────────────────


class TestDualWrite:
    def test_create_exists_in_both_stores(self, store: Store):
        obj_id = _create_sample(store, title="Dual write")

        # SQLite side
        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["title"] == "Dual write"

        # Graph side
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj["title"] == "Dual write"

    def test_read_returns_content_with_relationships(
        self, store: Store
    ):
        id_a = _create_sample(store, title="A")
        id_b = _create_sample(store, title="B")
        store.create_relationship(
            from_id=id_a, rel_type="supports", to_id=id_b
        )

        result = store.read(id_a)
        assert result is not None
        assert result["title"] == "A"
        assert "relationships" in result
        rels = result["relationships"]
        assert any(
            r["rel_type"] == "supports" and r["other_id"] == id_b
            for r in rels
        )

    def test_update_propagates_to_both_stores(self, store: Store):
        obj_id = _create_sample(store, title="Before")
        store.update(obj_id, title="After")

        # SQLite updated
        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["title"] == "After"

        # Graph updated
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj["title"] == "After"

    def test_update_nonexistent_raises(self, store: Store):
        with pytest.raises(NotFoundError):
            store.update("nonexistent-id", title="nope")

    def test_delete_removes_from_both_stores(self, store: Store):
        obj_id = _create_sample(store)
        assert store.delete(obj_id) is True

        assert store.content.get(obj_id) is None
        assert store.graph.read_object(obj_id) is None

    def test_delete_nonexistent_returns_false(self, store: Store):
        assert store.delete("no-such-id") is False


# ── Search & list ────────────────────────────────────────────────────


class TestSearchAndList:
    def test_search_returns_fts5_results(self, store: Store):
        _create_sample(store, title="Quantum entanglement")
        _create_sample(store, title="Classical mechanics")

        results = store.search("quantum")
        assert len(results) == 1
        assert results[0]["title"] == "Quantum entanglement"

    def test_list_objects_unfiltered(self, store: Store):
        _create_sample(store, title="A", obj_type="fix")
        _create_sample(store, title="B", obj_type="lesson")
        objects = store.list_objects()
        assert len(objects) == 2

    def test_list_objects_filter_by_type(self, store: Store):
        _create_sample(store, title="A", obj_type="fix")
        _create_sample(store, title="B", obj_type="lesson")
        objects = store.list_objects(obj_type="fix")
        assert len(objects) == 1

    def test_list_objects_filter_by_project(self, store: Store):
        _create_sample(store, title="A", project="alpha")
        _create_sample(store, title="B", project="beta")
        objects = store.list_objects(project="beta")
        assert len(objects) == 1


# ── Relationships ────────────────────────────────────────────────────


class TestRelationships:
    def test_create_and_get_relationships(self, store: Store):
        id_a = _create_sample(store, title="Cause")
        id_b = _create_sample(store, title="Effect")

        assert store.create_relationship(
            from_id=id_a, rel_type="causedBy", to_id=id_b
        )

        rels = store.get_relationships(id_a)
        assert any(
            r["rel_type"] == "causedBy"
            and r["other_id"] == id_b
            and r["direction"] == "outgoing"
            for r in rels
        )

        # Incoming side
        rels_b = store.get_relationships(id_b)
        assert any(
            r["rel_type"] == "causedBy"
            and r["other_id"] == id_a
            and r["direction"] == "incoming"
            for r in rels_b
        )

    def test_delete_relationship(self, store: Store):
        id_a = _create_sample(store, title="X")
        id_b = _create_sample(store, title="Y")
        store.create_relationship(
            from_id=id_a, rel_type="supports", to_id=id_b
        )
        assert store.delete_relationship(
            from_id=id_a, rel_type="supports", to_id=id_b
        )
        rels = store.get_relationships(id_a)
        assert not any(r["rel_type"] == "supports" for r in rels)


# ── Entities ─────────────────────────────────────────────────────────


class TestEntities:
    def test_create_entity_and_list(self, store: Store):
        eid = store.create_entity(
            name="Python", entity_type="technology"
        )
        assert eid

        entities = store.list_entities()
        assert any(e["name"] == "Python" for e in entities)

    def test_add_mention_links_object_to_entity(self, store: Store):
        obj_id = _create_sample(store, title="Python guide")
        eid = store.create_entity(
            name="Python", entity_type="technology"
        )
        store.add_mention(obj_id=obj_id, entity_id=eid)

        mentions = store.graph.get_entity_mentions(eid)
        assert obj_id in mentions

    def test_list_entities_filtered_by_type(self, store: Store):
        store.create_entity(name="Go", entity_type="technology")
        store.create_entity(name="CQRS", entity_type="pattern")

        techs = store.list_entities(entity_type="technology")
        assert all(e["type"] == "technology" for e in techs)
        assert any(e["name"] == "Go" for e in techs)


# ── Status ───────────────────────────────────────────────────────────


class TestStatus:
    def test_status_returns_counts(self, store: Store):
        _create_sample(store, obj_type="fix")
        _create_sample(store, obj_type="lesson")

        status = store.status()
        assert status["initialized"] is True
        assert status["sqlite_total"] == 2
        assert status["graph_triples"] > 0
        assert "fix" in status["counts_by_type"]
        assert "lesson" in status["counts_by_type"]


# ── Cross-system consistency ─────────────────────────────────────────


class TestCrossSystemConsistency:
    def test_ids_and_types_match_after_mixed_operations(
        self, store: Store
    ):
        """After 20 mixed ops, IDs and types stay in sync."""
        created_ids: list[str] = []
        types = [
            "decision", "lesson", "fix", "session",
            "research", "source", "synthesis", "idea",
        ]

        # 16 creates
        for i in range(16):
            t = types[i % len(types)]
            obj_id = store.create(
                obj_type=t,
                title=f"Object {i}",
                content=f"Content {i}",
                project="consistency",
                tags=t,
            )
            created_ids.append(obj_id)

        # 2 updates
        store.update(created_ids[0], title="Updated 0")
        store.update(created_ids[1], title="Updated 1")

        # 2 deletes
        store.delete(created_ids[14])
        store.delete(created_ids[15])

        # That is 20 operations: 16 + 2 + 2

        # Remaining IDs from SQLite
        remaining = created_ids[:14]
        for obj_id in remaining:
            sqlite_doc = store.content.get(obj_id)
            graph_obj = store.graph.read_object(obj_id)
            assert sqlite_doc is not None, (
                f"Missing from SQLite: {obj_id}"
            )
            assert graph_obj is not None, (
                f"Missing from graph: {obj_id}"
            )
            # Types must match (graph lowercases the class name)
            assert sqlite_doc["type"] == graph_obj["type"]

        # Deleted IDs are gone from both
        for obj_id in [created_ids[14], created_ids[15]]:
            assert store.content.get(obj_id) is None
            assert store.graph.read_object(obj_id) is None
