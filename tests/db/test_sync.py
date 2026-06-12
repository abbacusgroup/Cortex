"""Tests for cortex.db.store (unified dual-write Store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.core.errors import NotFoundError, StoreError, SyncError, ValidationError
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

    def test_read_returns_content_with_relationships(self, store: Store):
        id_a = _create_sample(store, title="A")
        id_b = _create_sample(store, title="B")
        store.create_relationship(from_id=id_a, rel_type="supports", to_id=id_b)

        result = store.read(id_a)
        assert result is not None
        assert result["title"] == "A"
        assert "relationships" in result
        rels = result["relationships"]
        assert any(r["rel_type"] == "supports" and r["other_id"] == id_b for r in rels)

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


# ── Short-id resolution & existence ──────────────────────────────────


class TestResolveIdAndExists:
    def test_exists_true_for_created_object(self, store: Store):
        obj_id = _create_sample(store)
        assert store.exists(obj_id) is True

    def test_exists_false_for_unknown_id(self, store: Store):
        assert store.exists("totally-bogus-id-12345") is False

    def test_resolve_full_id_unchanged(self, store: Store):
        obj_id = _create_sample(store)
        assert store.resolve_id(obj_id) == obj_id

    def test_resolve_unique_short_prefix(self, store: Store):
        obj_id = _create_sample(store)
        assert store.resolve_id(obj_id[:8]) == obj_id

    def test_resolve_unknown_returns_input_unchanged(self, store: Store):
        _create_sample(store)
        assert store.resolve_id("zzzzzzzz") == "zzzzzzzz"

    def test_resolve_empty_returns_input_unchanged(self, store: Store):
        assert store.resolve_id("") == ""

    def test_resolve_ambiguous_prefix_raises(self, store: Store):
        store.content.insert(
            doc_id="aaaa1111-0000-0000-0000-000000000000", title="A"
        )
        store.content.insert(
            doc_id="aaaa2222-0000-0000-0000-000000000000", title="B"
        )
        with pytest.raises(ValidationError) as exc:
            store.resolve_id("aaaa")
        assert set(exc.value.context["candidates"]) == {
            "aaaa1111-0000-0000-0000-000000000000",
            "aaaa2222-0000-0000-0000-000000000000",
        }

    def test_exact_match_wins_over_prefix(self, store: Store):
        # Imported ids can be arbitrary strings — one id may be a strict
        # prefix of another. The exact match must win, not raise ambiguity.
        store.content.insert(doc_id="abc", title="Exact")
        store.content.insert(doc_id="abcd", title="Longer")
        assert store.resolve_id("abc") == "abc"


# ── Relationships ────────────────────────────────────────────────────


class TestRelationships:
    def test_create_and_get_relationships(self, store: Store):
        id_a = _create_sample(store, title="Cause")
        id_b = _create_sample(store, title="Effect")

        assert store.create_relationship(from_id=id_a, rel_type="causedBy", to_id=id_b)

        rels = store.get_relationships(id_a)
        assert any(
            r["rel_type"] == "causedBy" and r["other_id"] == id_b and r["direction"] == "outgoing"
            for r in rels
        )

        # Incoming side
        rels_b = store.get_relationships(id_b)
        assert any(
            r["rel_type"] == "causedBy" and r["other_id"] == id_a and r["direction"] == "incoming"
            for r in rels_b
        )

    def test_delete_relationship(self, store: Store):
        id_a = _create_sample(store, title="X")
        id_b = _create_sample(store, title="Y")
        store.create_relationship(from_id=id_a, rel_type="supports", to_id=id_b)
        assert store.delete_relationship(from_id=id_a, rel_type="supports", to_id=id_b)
        rels = store.get_relationships(id_a)
        assert not any(r["rel_type"] == "supports" for r in rels)


# ── Entities ─────────────────────────────────────────────────────────


class TestEntities:
    def test_create_entity_and_list(self, store: Store):
        eid, _ = store.create_entity(name="Python", entity_type="technology")
        assert eid

        entities = store.list_entities()
        assert any(e["name"] == "Python" for e in entities)

    def test_add_mention_links_object_to_entity(self, store: Store):
        obj_id = _create_sample(store, title="Python guide")
        eid, _ = store.create_entity(name="Python", entity_type="technology")
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
    def test_ids_and_types_match_after_mixed_operations(self, store: Store):
        """After 20 mixed ops, IDs and types stay in sync."""
        created_ids: list[str] = []
        types = [
            "decision",
            "lesson",
            "fix",
            "session",
            "research",
            "source",
            "synthesis",
            "idea",
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
            assert sqlite_doc is not None, f"Missing from SQLite: {obj_id}"
            assert graph_obj is not None, f"Missing from graph: {obj_id}"
            # Types must match (graph lowercases the class name)
            assert sqlite_doc["type"] == graph_obj["type"]

        # Deleted IDs are gone from both
        for obj_id in [created_ids[14], created_ids[15]]:
            assert store.content.get(obj_id) is None
            assert store.graph.read_object(obj_id) is None


# ── Update key mapping (column → graph predicate translation) ────────


class TestUpdateKeyMapping:
    """Store.update must translate SQLite column names into graph predicates.

    Regression for the dual-write drift bug: column names were forwarded
    verbatim, so ``type`` became a literal cortex:type triple (the class
    never changed) and ``captured_by`` diverged from creation's capturedBy.
    """

    def test_type_change_rewrites_graph_class(self, store: Store):
        obj_id = _create_sample(store, obj_type="idea")
        store.update(obj_id, type="lesson")

        doc = store.content.get(obj_id)
        assert doc is not None and doc["type"] == "lesson"

        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj["type"] == "lesson"

    def test_type_change_keeps_per_type_counts_consistent(self, store: Store):
        obj_id = _create_sample(store, obj_type="idea")
        store.update(obj_id, type="lesson")

        status = store.status()
        assert status["counts_by_type"] == status["graph_counts_by_type"]
        assert status["counts_by_type"] == {"lesson": 1}

    def test_type_change_leaves_no_literal_type_triple(self, store: Store):
        from cortex.ontology.namespaces import cortex_iri

        obj_id = _create_sample(store, obj_type="idea")
        store.update(obj_id, type="lesson")

        subject = cortex_iri(f"obj/{obj_id}")
        stray = list(store.graph._store.quads_for_pattern(subject, cortex_iri("type"), None))
        assert stray == []

    def test_type_change_keeps_base_class_listable(self, store: Store):
        obj_id = _create_sample(store, obj_type="idea")
        store.update(obj_id, type="fix")

        objs = store.graph.list_objects()
        assert [o["id"] for o in objs] == [obj_id]
        assert objs[0]["type"] == "fix"

    def test_invalid_type_rejected_before_any_write(self, store: Store):
        obj_id = _create_sample(store, obj_type="idea")
        with pytest.raises(ValidationError):
            store.update(obj_id, type="nonsense")

        # Neither store was touched
        doc = store.content.get(obj_id)
        assert doc is not None and doc["type"] == "idea"
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None and graph_obj["type"] == "idea"

    def test_captured_by_maps_to_camelcase_predicate(self, store: Store):
        obj_id = _create_sample(store)
        store.update(obj_id, captured_by="claude")

        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj.get("capturedBy") == "claude"
        assert "captured_by" not in graph_obj

    def test_content_only_columns_not_forwarded_to_graph(self, store: Store):
        obj_id = _create_sample(store)
        store.update(
            obj_id, raw_markdown="# raw", pipeline_stage="enriched", confidence=0.5
        )

        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert "raw_markdown" not in graph_obj
        assert "pipeline_stage" not in graph_obj
        # Graph confidence stays at the creation-time typed literal
        # (Oxigraph normalizes the xsd:float lexical form, e.g. "1.0" → "1")
        assert float(graph_obj.get("confidence")) == 1.0

        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["raw_markdown"] == "# raw"
        assert doc["pipeline_stage"] == "enriched"
        assert doc["confidence"] == 0.5

    def test_graph_only_properties_kwarg_updates_graph(self, store: Store):
        obj_id = _create_sample(store, obj_type="decision")
        store.update(obj_id, properties={"rationale": "because tests"})

        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj.get("rationale") == "because tests"
        # SQLite row has no such column and is untouched
        doc = store.content.get(obj_id)
        assert doc is not None and "rationale" not in doc

    def test_type_in_properties_is_ignored(self, store: Store):
        obj_id = _create_sample(store, obj_type="idea")
        store.update(obj_id, properties={"type": "lesson"})

        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None and graph_obj["type"] == "idea"
        doc = store.content.get(obj_id)
        assert doc is not None and doc["type"] == "idea"

    def test_stray_literal_type_triple_cleaned_on_reclassify(self, store: Store):
        """Artifacts of the old bug are swept when the type is next rewritten."""
        import pyoxigraph as ox

        from cortex.ontology.namespaces import cortex_iri

        obj_id = _create_sample(store, obj_type="idea")
        subject = cortex_iri(f"obj/{obj_id}")
        store.graph._store.add(ox.Quad(subject, cortex_iri("type"), ox.Literal("lesson")))

        store.update(obj_id, type="fix")

        stray = list(store.graph._store.quads_for_pattern(subject, cortex_iri("type"), None))
        assert stray == []
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None and graph_obj["type"] == "fix"


# ── Update failure paths ─────────────────────────────────────────────


class TestUpdateFailurePaths:
    def test_graph_failure_raises_sync_error(self, store: Store, monkeypatch):
        obj_id = _create_sample(store)

        def boom(*args, **kwargs):
            raise RuntimeError("graph down")

        monkeypatch.setattr(store.graph, "update_object", boom)
        with pytest.raises(SyncError):
            store.update(obj_id, title="New title")

    def test_content_failure_raises_sync_error(self, store: Store, monkeypatch):
        obj_id = _create_sample(store)

        def boom(*args, **kwargs):
            raise RuntimeError("sqlite down")

        monkeypatch.setattr(store.content, "update", boom)
        with pytest.raises(SyncError):
            store.update(obj_id, title="New title")

    def test_graph_not_found_logs_but_sqlite_update_succeeds(self, store: Store):
        obj_id = _create_sample(store)
        store.graph.delete_object(obj_id)  # induce divergence

        assert store.update(obj_id, title="Still updates SQLite") is True
        doc = store.content.get(obj_id)
        assert doc is not None and doc["title"] == "Still updates SQLite"


# ── Delete safety ────────────────────────────────────────────────────


class TestDeleteSafety:
    def test_delete_snapshots_version_history(self, store: Store):
        obj_id = _create_sample(store, title="Doomed")
        assert store.delete(obj_id) is True

        assert store.temporal is not None
        versions = store.temporal.list_versions(obj_id)
        assert len(versions) == 1
        snap = store.temporal.get_version(obj_id, versions[0]["version_num"])
        assert snap is not None and snap["title"] == "Doomed"

    def test_content_delete_failure_raises_sync_error_naming_graph(
        self, store: Store, monkeypatch
    ):
        obj_id = _create_sample(store)

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(store.content, "delete", boom)
        with pytest.raises(SyncError) as exc_info:
            store.delete(obj_id)
        assert "Graph delete succeeded" in str(exc_info.value)

    def test_graph_delete_failure_raises_sync_error_and_keeps_sqlite_row(
        self, store: Store, monkeypatch
    ):
        obj_id = _create_sample(store)

        def boom(*args, **kwargs):
            raise RuntimeError("rocksdb error")

        monkeypatch.setattr(store.graph, "delete_object", boom)
        with pytest.raises(SyncError):
            store.delete(obj_id)
        # SQLite row untouched — no half-delete in the wrong direction
        assert store.content.get(obj_id) is not None


# ── Previously-untested db-layer branches ────────────────────────────


class TestUntestedDbBranches:
    def test_count_entities(self, store: Store):
        store.create_entity(name="Python", entity_type="technology")
        store.create_entity(name="CQRS", entity_type="pattern")

        assert store.graph.count_entities() == 2
        assert store.status()["entities"] == 2

    def test_update_immutable_column_rejected(self, store: Store):
        obj_id = _create_sample(store)
        with pytest.raises(StoreError, match="immutable"):
            store.content.update(obj_id, created_at="2020-01-01T00:00:00+00:00")

    def test_update_unknown_column_rejected(self, store: Store):
        obj_id = _create_sample(store)
        with pytest.raises(StoreError, match="Unknown column"):
            store.content.update(obj_id, bogus="x")


# ── Per-type count consistency across a mutation sequence ────────────


class TestCountsConsistencyAfterMutationSequence:
    def test_classify_update_delete_sequence_keeps_counts_in_sync(self, store: Store):
        """The empirically-observed drift case: after reclassify + update +
        delete, SQLite and graph per-type counts must agree exactly."""
        ids = [_create_sample(store, obj_type="idea", title=f"I{i}") for i in range(3)]
        ids.append(_create_sample(store, obj_type="fix", title="F0"))

        # Reclassify (the cortex_classify path)
        store.update(ids[0], type="lesson", summary="now a lesson", confidence=0.9)
        # Plain metadata update
        store.update(ids[1], title="Renamed", tags="x,y")
        # Delete
        store.delete(ids[3])

        status = store.status()
        assert status["counts_by_type"] == status["graph_counts_by_type"]
        assert status["counts_by_type"] == {"idea": 2, "lesson": 1}


class TestTimestamps:
    def test_create_with_custom_timestamps_reaches_both_stores(self, store):
        ts = "2026-03-23T00:00:00+00:00"
        obj_id = store.create(
            obj_type="fix",
            title="Timestamp threading test",
            content="Verifying timestamps reach both stores",
            created_at=ts,
            updated_at=ts,
        )
        # SQLite should have the custom timestamp
        doc = store.content.get(obj_id)
        assert doc is not None
        assert doc["created_at"] == ts
        assert doc["updated_at"] == ts

        # Graph should have the custom timestamp (Oxigraph normalizes +00:00 to Z)
        graph_obj = store.graph.read_object(obj_id)
        assert graph_obj is not None
        assert graph_obj["capturedAt"] == "2026-03-23T00:00:00Z"
