"""Tests for cortex.retrieval.graph (GraphQueries)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.retrieval.graph import GraphQueries

ONTOLOGY_PATH = find_ontology()


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Initialized Store backed by tmp_path."""
    cfg = CortexConfig(data_dir=tmp_path)
    s = Store(cfg)
    s.initialize(ONTOLOGY_PATH)
    return s


@pytest.fixture()
def gq(store: Store) -> GraphQueries:
    return GraphQueries(store)


def _obj(
    store: Store,
    title: str = "Obj",
    obj_type: str = "decision",
    project: str = "test",
) -> str:
    return store.create(
        obj_type=obj_type,
        title=title,
        content=title,
        project=project,
    )


# ── Causal Chain ────────────────────────────────────────────────────


class TestCausalChain:
    def test_caused_by_includes_both(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """A causedBy B -> chain starting from A includes both."""
        id_a = _obj(store, title="Effect")
        id_b = _obj(store, title="Cause")
        store.create_relationship(
            from_id=id_a,
            rel_type="causedBy",
            to_id=id_b,
        )

        chain = gq.causal_chain(id_a)
        chain_ids = [c["id"] for c in chain]
        assert id_a in chain_ids
        assert id_b in chain_ids

    def test_no_causal_relations_returns_self(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """Object with no causal edges -> chain has just itself."""
        obj_id = _obj(store, title="Lonely")

        chain = gq.causal_chain(obj_id)
        assert len(chain) == 1
        assert chain[0]["id"] == obj_id

    def test_chain_terminates_at_max_depth(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """Chain longer than max_depth stops early."""
        ids = [_obj(store, title=f"Node-{i}") for i in range(5)]
        for i in range(4):
            store.create_relationship(
                from_id=ids[i],
                rel_type="causedBy",
                to_id=ids[i + 1],
            )

        chain = gq.causal_chain(ids[0], max_depth=2)
        # Should not traverse the full depth-4 chain
        assert len(chain) <= 4

    def test_circular_chain_terminates(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """Circular causal edges do not cause infinite recursion."""
        id_a = _obj(store, title="A")
        id_b = _obj(store, title="B")
        id_c = _obj(store, title="C")
        store.create_relationship(
            from_id=id_a,
            rel_type="causedBy",
            to_id=id_b,
        )
        store.create_relationship(
            from_id=id_b,
            rel_type="causedBy",
            to_id=id_c,
        )
        store.create_relationship(
            from_id=id_c,
            rel_type="causedBy",
            to_id=id_a,
        )

        # Must return without hanging; visited set breaks the cycle
        chain = gq.causal_chain(id_a)
        assert isinstance(chain, list)
        chain_ids = [c["id"] for c in chain]
        assert id_a in chain_ids


# ── Contradiction Map ───────────────────────────────────────────────


class TestContradictionMap:
    def test_contradiction_appears_in_map(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        id_a = _obj(store, title="Claim A")
        id_b = _obj(store, title="Claim B")
        store.create_relationship(
            from_id=id_a,
            rel_type="contradicts",
            to_id=id_b,
        )

        result = gq.contradiction_map()
        assert len(result) >= 1
        pair = result[0]
        assert "object_a" in pair
        assert "object_b" in pair
        pair_ids = {pair["object_a"], pair["object_b"]}
        assert pair_ids == {id_a, id_b}

    def test_no_contradictions_returns_empty(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        _obj(store, title="Peaceful A")
        _obj(store, title="Peaceful B")

        assert gq.contradiction_map() == []

    def test_scoped_by_project(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """Contradictions in project 'alpha' don't appear in 'beta'."""
        id_a = _obj(store, title="Alpha A", project="alpha")
        id_b = _obj(store, title="Alpha B", project="alpha")
        store.create_relationship(
            from_id=id_a,
            rel_type="contradicts",
            to_id=id_b,
        )
        _obj(store, title="Beta C", project="beta")

        alpha_map = gq.contradiction_map(scope="alpha")
        assert len(alpha_map) >= 1

        beta_map = gq.contradiction_map(scope="beta")
        assert len(beta_map) == 0


# ── Entity Neighborhood ────────────────────────────────────────────


class TestEntityNeighborhood:
    def test_entity_with_mentions(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """Entity mentioned by objects -> objects returned."""
        eid, _ = store.create_entity(
            name="Python",
            entity_type="technology",
        )
        obj_id = _obj(store, title="Python guide")
        store.add_mention(obj_id=obj_id, entity_id=eid)

        result = gq.entity_neighborhood("Python")
        assert result["entity"] is not None
        assert result["entity"]["name"] == "Python"
        obj_ids = [o["id"] for o in result["objects"]]
        assert obj_id in obj_ids

    def test_two_hop_connections(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """2-hop: entity -> obj_a -> obj_b via relationship."""
        eid, _ = store.create_entity(
            name="Docker",
            entity_type="technology",
        )
        id_a = _obj(store, title="Docker setup")
        id_b = _obj(store, title="K8s migration")
        store.add_mention(obj_id=id_a, entity_id=eid)
        store.create_relationship(
            from_id=id_a,
            rel_type="ledTo",
            to_id=id_b,
        )

        result = gq.entity_neighborhood("Docker", max_hops=2)
        conn_ids = [c["id"] for c in result["connections"]]
        assert id_b in conn_ids

    def test_unknown_entity_returns_empty(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        result = gq.entity_neighborhood("Nonexistent")
        assert result["entity"] is None
        assert result["objects"] == []
        assert result["connections"] == []


# ── Evolution Timeline ──────────────────────────────────────────────


class TestEvolutionTimeline:
    def test_supersedes_order(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """A supersedes B -> timeline has B then A (oldest first)."""
        id_old = _obj(store, title="v1")
        id_new = _obj(store, title="v2")
        # "v2 supersedes v1"
        store.create_relationship(
            from_id=id_new,
            rel_type="supersedes",
            to_id=id_old,
        )

        timeline = gq.evolution_timeline(id_new)
        ids = [t["id"] for t in timeline]
        assert ids.index(id_old) < ids.index(id_new)

    def test_single_object_timeline(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        obj_id = _obj(store, title="Standalone")

        timeline = gq.evolution_timeline(obj_id)
        assert len(timeline) == 1
        assert timeline[0]["id"] == obj_id

    def test_three_step_chain(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """C supersedes B supersedes A -> timeline [A, B, C]."""
        id_a = _obj(store, title="v1")
        id_b = _obj(store, title="v2")
        id_c = _obj(store, title="v3")
        store.create_relationship(
            from_id=id_b,
            rel_type="supersedes",
            to_id=id_a,
        )
        store.create_relationship(
            from_id=id_c,
            rel_type="supersedes",
            to_id=id_b,
        )

        timeline = gq.evolution_timeline(id_c)
        ids = [t["id"] for t in timeline]
        assert ids == [id_a, id_b, id_c]


# ── Project Overview ────────────────────────────────────────────────


class TestProjectOverview:
    def test_returns_objects_and_edges(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        id_a = _obj(store, title="Obj A", project="proj")
        id_b = _obj(store, title="Obj B", project="proj")
        store.create_relationship(
            from_id=id_a,
            rel_type="supports",
            to_id=id_b,
        )

        overview = gq.project_overview("proj")
        assert overview["project"] == "proj"
        assert overview["object_count"] == 2
        assert len(overview["objects"]) == 2
        assert len(overview["edges"]) >= 1
        assert any(e["from"] == id_a and e["to"] == id_b for e in overview["edges"])

    def test_project_entities_found_via_mentions(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        """project_overview finds entities via get_entity_mentions."""
        obj_id = _obj(store, title="Go service", project="ent-proj")
        ent_id, _ = store.create_entity(
            name="Go",
            entity_type="technology",
        )
        store.add_mention(obj_id=obj_id, entity_id=ent_id)

        overview = gq.project_overview("ent-proj")
        assert len(overview["entities"]) == 1
        assert overview["entities"][0]["name"] == "Go"

    def test_empty_project(
        self,
        store: Store,
        gq: GraphQueries,
    ):
        overview = gq.project_overview("empty")
        assert overview["object_count"] == 0
        assert overview["objects"] == []
        assert overview["entities"] == []
        assert overview["edges"] == []
