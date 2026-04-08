"""Tests for the GraphStore Oxigraph-backed RDF store.

Covers CRUD for knowledge objects, relationships, entities,
and SPARQL queries — including adversarial edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.errors import NotFoundError, StoreError, ValidationError
from cortex.db.graph_store import GraphStore
from cortex.ontology.resolver import find_ontology

ONTOLOGY_PATH = find_ontology()

OBJECT_TYPES = [
    "decision",
    "lesson",
    "fix",
    "session",
    "research",
    "source",
    "synthesis",
    "idea",
]


@pytest.fixture()
def store() -> GraphStore:
    """In-memory GraphStore with ontology loaded."""
    gs = GraphStore(path=None)
    gs.load_ontology(ONTOLOGY_PATH)
    return gs


# =========================================================================
# CRUD — Happy Path
# =========================================================================


class TestCreateRead:
    """create_object / read_object round-trip."""

    def test_create_returns_uuid(self, store: GraphStore) -> None:
        obj_id = store.create_object(
            obj_type="decision",
            title="Use Oxigraph",
            content="It is fast.",
        )
        assert obj_id
        assert len(obj_id) == 36  # UUID format

    def test_read_returns_all_properties(self, store: GraphStore) -> None:
        obj_id = store.create_object(
            obj_type="fix",
            title="Null pointer fix",
            content="Check for None first.",
            project="cortex",
            tags="bug,urgent",
            tier="recall",
            captured_by="test",
            confidence=0.95,
            properties={"symptom": "crash", "rootCause": "null ref"},
        )
        obj = store.read_object(obj_id)
        assert obj is not None
        assert obj["id"] == obj_id
        assert obj["type"] == "fix"
        assert obj["title"] == "Null pointer fix"
        assert obj["content"] == "Check for None first."
        assert obj["project"] == "cortex"
        assert obj["tags"] == "bug,urgent"
        assert obj["tier"] == "recall"
        assert obj["capturedBy"] == "test"
        assert obj["symptom"] == "crash"
        assert obj["rootCause"] == "null ref"

    @pytest.mark.parametrize("obj_type", OBJECT_TYPES)
    def test_create_all_types(
        self, store: GraphStore, obj_type: str
    ) -> None:
        obj_id = store.create_object(
            obj_type=obj_type, title=f"Test {obj_type}"
        )
        obj = store.read_object(obj_id)
        assert obj is not None
        assert obj["type"] == obj_type

    def test_create_with_custom_captured_at(
        self, store: GraphStore
    ) -> None:
        ts = "2026-03-23T00:00:00+00:00"
        obj_id = store.create_object(
            obj_type="decision",
            title="Backdated decision",
            captured_at=ts,
        )
        obj = store.read_object(obj_id)
        assert obj is not None
        # Oxigraph normalises "+00:00" → "Z"
        assert obj["capturedAt"] == "2026-03-23T00:00:00Z"


class TestUpdate:
    """update_object modifies properties."""

    def test_update_changes_property(self, store: GraphStore) -> None:
        obj_id = store.create_object(
            obj_type="lesson", title="Original title"
        )
        store.update_object(obj_id, title="Updated title")
        obj = store.read_object(obj_id)
        assert obj is not None
        assert obj["title"] == "Updated title"

    def test_update_adds_new_property(self, store: GraphStore) -> None:
        obj_id = store.create_object(
            obj_type="idea", title="New idea"
        )
        store.update_object(obj_id, feasibility="high")
        obj = store.read_object(obj_id)
        assert obj is not None
        assert obj["feasibility"] == "high"


class TestDelete:
    """delete_object removes all triples."""

    def test_delete_removes_object(self, store: GraphStore) -> None:
        obj_id = store.create_object(
            obj_type="research", title="Temp"
        )
        assert store.delete_object(obj_id) is True
        assert store.read_object(obj_id) is None

    def test_delete_removes_incoming_relationships(
        self, store: GraphStore
    ) -> None:
        a = store.create_object(obj_type="decision", title="A")
        b = store.create_object(obj_type="lesson", title="B")
        store.create_relationship(
            from_id=a, rel_type="supports", to_id=b
        )
        store.delete_object(b)
        rels = store.get_relationships(a)
        assert len(rels) == 0


class TestListObjects:
    """list_objects with filters."""

    def test_list_by_type(self, store: GraphStore) -> None:
        store.create_object(
            obj_type="fix", title="Fix1", project="p1"
        )
        store.create_object(
            obj_type="lesson", title="Lesson1", project="p1"
        )
        fixes = store.list_objects(obj_type="fix")
        assert len(fixes) == 1
        assert fixes[0]["type"] == "fix"

    def test_list_by_project(self, store: GraphStore) -> None:
        store.create_object(
            obj_type="fix", title="Fix1", project="alpha"
        )
        store.create_object(
            obj_type="fix", title="Fix2", project="beta"
        )
        alpha = store.list_objects(project="alpha")
        assert len(alpha) == 1
        assert alpha[0]["project"] == "alpha"

    def test_list_empty(self, store: GraphStore) -> None:
        result = store.list_objects(obj_type="decision")
        assert result == []


# =========================================================================
# CRUD — Edge Cases
# =========================================================================


class TestCrudEdgeCases:
    """Boundary conditions and adversarial inputs for CRUD."""

    def test_unicode_title_roundtrip(self, store: GraphStore) -> None:
        title = "Deploy to production \U0001f680 \u6d4b\u8bd5"
        obj_id = store.create_object(obj_type="session", title=title)
        obj = store.read_object(obj_id)
        assert obj is not None
        assert obj["title"] == title

    def test_update_nonexistent_raises_not_found(
        self, store: GraphStore
    ) -> None:
        with pytest.raises(NotFoundError):
            store.update_object(
                "00000000-0000-0000-0000-000000000000", title="nope"
            )

    def test_delete_nonexistent_returns_false(
        self, store: GraphStore
    ) -> None:
        result = store.delete_object(
            "00000000-0000-0000-0000-000000000000"
        )
        assert result is False

    def test_read_nonexistent_returns_none(
        self, store: GraphStore
    ) -> None:
        assert store.read_object(
            "00000000-0000-0000-0000-000000000000"
        ) is None

    def test_duplicate_titles_different_ids(
        self, store: GraphStore
    ) -> None:
        id1 = store.create_object(obj_type="idea", title="Same Title")
        id2 = store.create_object(obj_type="idea", title="Same Title")
        assert id1 != id2
        assert store.read_object(id1) is not None
        assert store.read_object(id2) is not None

    def test_sparql_injection_in_title(self, store: GraphStore) -> None:
        """SPARQL metacharacters in title are stored as literal text."""
        evil = 'Test" } DELETE WHERE { ?s ?p ?o } #'
        obj_id = store.create_object(obj_type="fix", title=evil)
        obj = store.read_object(obj_id)
        assert obj is not None
        assert obj["title"] == evil

    def test_content_exceeding_10mb_raises_validation(
        self, store: GraphStore
    ) -> None:
        big = "x" * (10 * 1024 * 1024 + 1)
        with pytest.raises(ValidationError, match="10MB"):
            store.create_object(
                obj_type="research", title="Big", content=big
            )

    def test_invalid_type_raises_validation(
        self, store: GraphStore
    ) -> None:
        with pytest.raises(ValidationError, match="Invalid knowledge type"):
            store.create_object(obj_type="foobar", title="Bad type")


# =========================================================================
# SPARQL literal escaping (Bundle 10.1 hardening)
# =========================================================================


class TestSparqlEscape:
    """Unit tests for the ``_sparql_escape_string`` helper.

    Pyoxigraph 0.5.x does not provide a parameterized query API, so any
    user-controllable value interpolated into a SPARQL string literal must
    pass through this helper. See ``_sparql_escape_string`` in
    ``cortex/db/graph_store.py`` and the SPARQL 1.1 spec §5.4 (``ECHAR``).
    """

    def test_clean_string_is_unchanged(self) -> None:
        from cortex.db.graph_store import _sparql_escape_string

        assert _sparql_escape_string("plain") == "plain"
        assert _sparql_escape_string("a b c") == "a b c"
        assert _sparql_escape_string("unicode \u6d4b") == "unicode \u6d4b"

    def test_backslash_is_escaped_first(self) -> None:
        from cortex.db.graph_store import _sparql_escape_string

        # A single backslash must become two, not four (order-dependency
        # regression guard: if we escape " before \, the \" insertion
        # would then get its own \ re-escaped).
        assert _sparql_escape_string("\\") == "\\\\"
        assert _sparql_escape_string("a\\b") == "a\\\\b"

    def test_double_quote_is_escaped(self) -> None:
        from cortex.db.graph_store import _sparql_escape_string

        assert _sparql_escape_string('"') == '\\"'
        assert _sparql_escape_string('say "hi"') == 'say \\"hi\\"'

    def test_whitespace_controls_are_escaped(self) -> None:
        from cortex.db.graph_store import _sparql_escape_string

        assert _sparql_escape_string("a\nb") == "a\\nb"
        assert _sparql_escape_string("a\rb") == "a\\rb"
        assert _sparql_escape_string("a\tb") == "a\\tb"

    def test_backslash_then_quote_does_not_double_escape(self) -> None:
        from cortex.db.graph_store import _sparql_escape_string

        # Literal input: \  "   →  \\  \"
        assert _sparql_escape_string('\\"') == '\\\\\\"'


class TestSparqlInjectionListObjects:
    """Adversarial tests for ``GraphStore.list_objects(project=...)``.

    Before Bundle 10.1 the ``project`` filter was interpolated directly
    into the SPARQL query. Pyoxigraph's lenient parser prevented actual
    exploitation but the underlying string was unescaped. These tests
    verify that the escape helper is applied correctly and that the
    filter behaves normally for hostile values.
    """

    def test_project_with_double_quote_filters_correctly(
        self, store: GraphStore
    ) -> None:
        evil_project = 'proj"name'
        store.create_object(obj_type="fix", title="F1", project=evil_project)
        store.create_object(obj_type="fix", title="F2", project="other")

        results = store.list_objects(project=evil_project)
        assert len(results) == 1
        assert results[0]["title"] == "F1"
        assert results[0]["project"] == evil_project

    def test_project_with_backslash_filters_correctly(
        self, store: GraphStore
    ) -> None:
        evil_project = "path\\to\\project"
        store.create_object(obj_type="fix", title="B1", project=evil_project)
        store.create_object(obj_type="fix", title="B2", project="clean")

        results = store.list_objects(project=evil_project)
        assert len(results) == 1
        assert results[0]["title"] == "B1"
        assert results[0]["project"] == evil_project

    def test_project_with_newline_filters_correctly(
        self, store: GraphStore
    ) -> None:
        evil_project = "line1\nline2"
        store.create_object(obj_type="fix", title="N1", project=evil_project)
        store.create_object(obj_type="fix", title="N2", project="plain")

        results = store.list_objects(project=evil_project)
        assert len(results) == 1
        assert results[0]["title"] == "N1"

    def test_project_injection_attempt_does_not_leak_other_rows(
        self, store: GraphStore
    ) -> None:
        # Classic injection payload: try to close the literal and append
        # a clause that would match everything. With the escape in place
        # the whole string is a literal, so nothing matches — the query
        # returns zero rows instead of leaking the "other" project.
        store.create_object(obj_type="fix", title="Hidden", project="other")
        store.create_object(obj_type="fix", title="Visible", project="target")

        payload = 'target" } UNION { ?s ?p ?o . FILTER(true) '
        results = store.list_objects(project=payload)
        assert results == []

        # Sanity check: the clean value still matches.
        assert len(store.list_objects(project="target")) == 1


# =========================================================================
# Relationships
# =========================================================================


class TestRelationships:
    """Relationship CRUD between knowledge objects."""

    def test_create_and_get_relationship(
        self, store: GraphStore
    ) -> None:
        a = store.create_object(obj_type="decision", title="A")
        b = store.create_object(obj_type="lesson", title="B")
        store.create_relationship(
            from_id=a, rel_type="causedBy", to_id=b
        )
        rels = store.get_relationships(a)
        assert any(
            r["rel_type"] == "causedBy" and r["other_id"] == b
            for r in rels
        )

    def test_delete_relationship(self, store: GraphStore) -> None:
        a = store.create_object(obj_type="fix", title="A")
        b = store.create_object(obj_type="fix", title="B")
        store.create_relationship(
            from_id=a, rel_type="supports", to_id=b
        )
        assert store.delete_relationship(
            from_id=a, rel_type="supports", to_id=b
        )
        rels = store.get_relationships(a)
        assert len(rels) == 0

    def test_self_referential_raises_validation(
        self, store: GraphStore
    ) -> None:
        a = store.create_object(obj_type="lesson", title="Self")
        with pytest.raises(ValidationError, match="Self-referential"):
            store.create_relationship(
                from_id=a, rel_type="contradicts", to_id=a
            )

    def test_duplicate_relationship_is_idempotent(
        self, store: GraphStore
    ) -> None:
        a = store.create_object(obj_type="decision", title="A")
        b = store.create_object(obj_type="decision", title="B")
        store.create_relationship(
            from_id=a, rel_type="supersedes", to_id=b
        )
        store.create_relationship(
            from_id=a, rel_type="supersedes", to_id=b
        )
        rels = [
            r
            for r in store.get_relationships(a)
            if r["rel_type"] == "supersedes"
        ]
        assert len(rels) == 1

    def test_delete_object_cleans_up_relationships(
        self, store: GraphStore
    ) -> None:
        a = store.create_object(obj_type="fix", title="A")
        b = store.create_object(obj_type="lesson", title="B")
        c = store.create_object(obj_type="session", title="C")
        store.create_relationship(
            from_id=a, rel_type="supports", to_id=b
        )
        store.create_relationship(
            from_id=c, rel_type="dependsOn", to_id=b
        )
        store.delete_object(b)
        assert store.get_relationships(a) == []
        assert store.get_relationships(c) == []

    def test_invalid_rel_type_raises_validation(
        self, store: GraphStore
    ) -> None:
        a = store.create_object(obj_type="idea", title="A")
        b = store.create_object(obj_type="idea", title="B")
        with pytest.raises(
            ValidationError, match="Invalid relationship type"
        ):
            store.create_relationship(
                from_id=a, rel_type="nonsense", to_id=b
            )

    def test_incoming_relationship_visible(
        self, store: GraphStore
    ) -> None:
        """The target should see the relationship as incoming."""
        a = store.create_object(obj_type="decision", title="A")
        b = store.create_object(obj_type="lesson", title="B")
        store.create_relationship(
            from_id=a, rel_type="ledTo", to_id=b
        )
        rels = store.get_relationships(b)
        incoming = [r for r in rels if r["direction"] == "incoming"]
        assert len(incoming) == 1
        assert incoming[0]["other_id"] == a


# =========================================================================
# Entities
# =========================================================================


class TestEntities:
    """Entity CRUD and mention linking."""

    def test_create_entity_returns_id(self, store: GraphStore) -> None:
        eid = store.create_entity(name="Python", entity_type="technology")
        assert eid
        assert len(eid) == 36

    def test_duplicate_name_case_insensitive_returns_existing(
        self, store: GraphStore
    ) -> None:
        id1 = store.create_entity(name="Oxigraph", entity_type="technology")
        id2 = store.create_entity(name="oxigraph", entity_type="technology")
        assert id1 == id2

    def test_list_entities_with_type_filter(
        self, store: GraphStore
    ) -> None:
        store.create_entity(name="Python", entity_type="technology")
        store.create_entity(name="SOLID", entity_type="pattern")
        techs = store.list_entities(entity_type="technology")
        assert len(techs) == 1
        assert techs[0]["name"] == "Python"

    def test_list_entities_all(self, store: GraphStore) -> None:
        store.create_entity(name="FastAPI", entity_type="technology")
        store.create_entity(name="DDD", entity_type="pattern")
        all_ents = store.list_entities()
        assert len(all_ents) == 2

    def test_add_mention_and_get_entity_mentions(
        self, store: GraphStore
    ) -> None:
        eid = store.create_entity(
            name="Oxigraph", entity_type="technology"
        )
        obj_id = store.create_object(
            obj_type="research", title="Graph DB research"
        )
        store.add_mention(obj_id=obj_id, entity_id=eid)
        mentions = store.get_entity_mentions(eid)
        assert obj_id in mentions

    def test_entity_with_unknown_type_defaults_to_concept(
        self, store: GraphStore
    ) -> None:
        eid = store.create_entity(
            name="Emergence", entity_type="bogus_type"
        )
        entities = store.list_entities(entity_type="concept")
        assert any(e["id"] == eid for e in entities)


# =========================================================================
# SPARQL
# =========================================================================


class TestSparql:
    """Direct SPARQL query interface."""

    def test_valid_query_returns_results(
        self, store: GraphStore
    ) -> None:
        store.create_object(obj_type="fix", title="SPARQL test fix")
        rows = store.query(
            "SELECT ?s ?title WHERE {"
            " ?s a cortex:Fix . ?s cortex:title ?title . }"
        )
        assert len(rows) >= 1
        assert any(r["title"] == "SPARQL test fix" for r in rows)

    def test_malformed_sparql_raises_store_error(
        self, store: GraphStore
    ) -> None:
        with pytest.raises(StoreError, match="SPARQL syntax"):
            store.query("SELECT WHERE {{ invalid sparql %%% }}")

    def test_count_by_type(self, store: GraphStore) -> None:
        store.create_object(obj_type="decision", title="D1")
        store.create_object(obj_type="decision", title="D2")
        store.create_object(obj_type="fix", title="F1")
        counts = store.count_by_type()
        assert counts.get("decision", 0) == 2
        assert counts.get("fix", 0) == 1

    def test_count_by_type_empty(self, store: GraphStore) -> None:
        counts = store.count_by_type()
        assert counts == {}

    def test_query_with_explicit_prefixes(
        self, store: GraphStore
    ) -> None:
        """Query that already has PREFIX declarations should not double-add."""
        store.create_object(obj_type="idea", title="Prefix test")
        rows = store.query(
            "PREFIX cortex: <https://cortex.abbacus.ai/ontology#>\n"
            "SELECT ?title WHERE {"
            " ?s a cortex:Idea . ?s cortex:title ?title . }"
        )
        assert len(rows) >= 1


# =========================================================================
# Persistence (tmp_path)
# =========================================================================


class TestPersistence:
    """Persistent store round-trip via tmp_path."""

    def test_persistent_store_roundtrip(self, tmp_path: Path) -> None:
        db_path = tmp_path / "graph.db"

        gs = GraphStore(path=db_path)
        gs.load_ontology(ONTOLOGY_PATH)
        obj_id = gs.create_object(
            obj_type="lesson", title="Persisted"
        )
        del gs  # close

        gs2 = GraphStore(path=db_path)
        obj = gs2.read_object(obj_id)
        assert obj is not None
        assert obj["title"] == "Persisted"
