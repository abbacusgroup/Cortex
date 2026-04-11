"""Benchmark: graph traversal intelligence.

Tests the five core GraphQueries traversals against hand-seeded graphs:
  1. Causal chain accuracy (causedBy/ledTo)
  2. Evolution timeline (supersedes chains)
  3. Entity neighborhood (mention + relationship hops)
  4. Project overview (objects, entities, edges)
  5. Contradiction map scoping (project-scoped contradicts)
"""

from __future__ import annotations

import pytest

from cortex.db.store import Store
from cortex.pipeline.reason import ReasonStage
from cortex.retrieval.graph import GraphQueries


# ── Helpers ───────────────────────────────────────────────────────────


def _ids(chain: list[dict]) -> list[str]:
    """Extract ordered list of IDs from a chain/timeline result."""
    return [item["id"] for item in chain]


# ── 1. Causal chain accuracy ─────────────────────────────────────────


@pytest.mark.bench
def test_causal_chain_accuracy(store: Store) -> None:
    """Causal chain follows causedBy backward and ledTo forward.

    Seed a 3-node chain A→B→C (B causedBy A, C causedBy B) then query
    from the MIDDLE node B. After ReasonStage generates ledTo inverses
    the chain should contain A, B, C in causal order — proving that
    forward traversal works from the query node.

    Also tests at depth 5 with a 5-node chain queried from the middle.
    """
    gq = GraphQueries(store)

    # -- 3-node chain --
    a_id = store.create(obj_type="decision", title="Decision A", content="Root cause", project="causal")
    b_id = store.create(obj_type="fix", title="Fix B", content="Middle fix", project="causal")
    c_id = store.create(obj_type="lesson", title="Lesson C", content="Outcome lesson", project="causal")

    store.create_relationship(from_id=b_id, rel_type="causedBy", to_id=a_id)
    store.create_relationship(from_id=c_id, rel_type="causedBy", to_id=b_id)

    # Generate ledTo inverses so forward traversal works
    ReasonStage(store.graph).run()

    # Query from the MIDDLE node — proves both backward and forward work
    chain = gq.causal_chain(b_id)
    ids = _ids(chain)

    assert len(chain) >= 3, f"Expected at least 3 nodes, got {len(chain)}"
    assert a_id in ids, "Root cause A must be in chain"
    assert b_id in ids, "Middle node B must be in chain"
    assert c_id in ids, "Effect C must be in chain"
    # Causal order: A before B before C
    assert ids.index(a_id) < ids.index(b_id), "A must precede B"
    assert ids.index(b_id) < ids.index(c_id), "B must precede C"

    # -- 5-node chain (depth test) --
    nodes = []
    for i in range(5):
        nid = store.create(
            obj_type="decision",
            title=f"Chain node {i}",
            content=f"Node {i} in deep chain",
            project="causal-deep",
        )
        nodes.append(nid)

    for i in range(1, 5):
        store.create_relationship(from_id=nodes[i], rel_type="causedBy", to_id=nodes[i - 1])

    ReasonStage(store.graph).run()

    # Query from the middle node (node 2) — full chain should be found
    deep_chain = gq.causal_chain(nodes[2], max_depth=5)
    deep_ids = _ids(deep_chain)

    assert len(deep_chain) >= 5, f"Expected at least 5 nodes, got {len(deep_chain)}"
    for node in nodes:
        assert node in deep_ids, f"Node {node} must be in deep chain"
    # Order preserved
    for i in range(4):
        assert deep_ids.index(nodes[i]) < deep_ids.index(nodes[i + 1]), (
            f"Node {i} must precede node {i + 1}"
        )


# ── 2. Evolution timeline ────────────────────────────────────────────


@pytest.mark.bench
def test_evolution_timeline(store: Store) -> None:
    """Supersedes chain produces a chronological evolution timeline.

    Seed v1→v2→v3→v4 (each supersedes its predecessor). Run ReasonStage
    to create transitive closure edges. Query from v4. The timeline
    should still be [v1, v2, v3, v4] thanks to created_at sorting.
    """
    gq = GraphQueries(store)

    v1 = store.create(obj_type="decision", title="Policy v1", content="Initial policy", project="evo")
    v2 = store.create(obj_type="decision", title="Policy v2", content="Revised policy", project="evo")
    v3 = store.create(obj_type="decision", title="Policy v3", content="Further revision", project="evo")
    v4 = store.create(obj_type="decision", title="Policy v4", content="Current policy", project="evo")

    # v2 supersedes v1, v3 supersedes v2, v4 supersedes v3
    store.create_relationship(from_id=v2, rel_type="supersedes", to_id=v1)
    store.create_relationship(from_id=v3, rel_type="supersedes", to_id=v2)
    store.create_relationship(from_id=v4, rel_type="supersedes", to_id=v3)

    # Transitive closure — previously broke ordering, now handled by sort
    ReasonStage(store.graph).run()

    timeline = gq.evolution_timeline(v4)
    ids = _ids(timeline)

    assert len(timeline) >= 4, f"Expected at least 4 versions, got {len(timeline)}"
    for v in (v1, v2, v3, v4):
        assert v in ids, f"Version {v} must be in timeline"

    # Chronological order: v1 < v2 < v3 < v4
    assert ids.index(v1) < ids.index(v2), "v1 must precede v2"
    assert ids.index(v2) < ids.index(v3), "v2 must precede v3"
    assert ids.index(v3) < ids.index(v4), "v3 must precede v4"


# ── 3. Entity neighborhood ───────────────────────────────────────────


@pytest.mark.bench
def test_entity_neighborhood(store: Store) -> None:
    """Entity neighborhood returns direct mentions (hop 1) and their
    relationship neighbors (hop 2).

    Seed entity "Redis", 5 fix objects mentioning it, and 3 additional
    objects connected to those fixes via "supports" relationships.
    """
    gq = GraphQueries(store)

    # Create entity
    redis_eid = store.create_entity(name="Redis", entity_type="technology")

    # Create 5 fix objects that mention Redis
    fix_ids = []
    for i in range(5):
        fid = store.create(
            obj_type="fix",
            title=f"Redis fix {i}",
            content=f"Fixed Redis connection issue #{i}",
            project="redis-bench",
        )
        store.add_mention(obj_id=fid, entity_id=redis_eid)
        fix_ids.append(fid)

    # Create 3 objects connected to fixes via "supports"
    support_ids = []
    for i in range(3):
        sid = store.create(
            obj_type="research",
            title=f"Redis research {i}",
            content=f"Research supporting Redis fix #{i}",
            project="redis-bench",
        )
        store.create_relationship(from_id=sid, rel_type="supports", to_id=fix_ids[i])
        support_ids.append(sid)

    result = gq.entity_neighborhood("Redis", max_hops=2)

    # Entity found
    assert result["entity"] is not None, "Entity 'Redis' must be found"
    assert result["entity"]["name"] == "Redis"

    # Hop 1: all 5 fixes
    hop1_ids = {obj["id"] for obj in result["objects"]}
    for fid in fix_ids:
        assert fid in hop1_ids, f"Fix {fid} must be in hop-1 objects"

    # Hop 2: the 3 connected research objects
    hop2_ids = {obj["id"] for obj in result["connections"]}
    for sid in support_ids:
        assert sid in hop2_ids, f"Support object {sid} must be in hop-2 connections"


# ── 4. Project overview ──────────────────────────────────────────────


@pytest.mark.bench
def test_project_overview(store: Store) -> None:
    """Project overview returns all objects, entities, and edges for a
    project.

    Seed 8 objects in project "gamma" with relationships and 3 entities
    mentioned by those objects.
    """
    gq = GraphQueries(store)

    # Create 8 objects of various types
    obj_types = ["decision", "fix", "lesson", "session", "research", "idea", "fix", "decision"]
    obj_ids = []
    for i, otype in enumerate(obj_types):
        oid = store.create(
            obj_type=otype,
            title=f"Gamma {otype} {i}",
            content=f"Content for gamma object {i}",
            project="gamma",
        )
        obj_ids.append(oid)

    # Wire some relationships
    store.create_relationship(from_id=obj_ids[1], rel_type="causedBy", to_id=obj_ids[0])
    store.create_relationship(from_id=obj_ids[2], rel_type="causedBy", to_id=obj_ids[1])
    store.create_relationship(from_id=obj_ids[4], rel_type="supports", to_id=obj_ids[0])
    store.create_relationship(from_id=obj_ids[7], rel_type="supersedes", to_id=obj_ids[0])
    store.create_relationship(from_id=obj_ids[5], rel_type="supports", to_id=obj_ids[3])

    # Create 3 entities and mention them from gamma objects
    entity_names = [("Postgres", "technology"), ("Auth", "concept"), ("K8s", "technology")]
    entity_ids = []
    for name, etype in entity_names:
        eid = store.create_entity(name=name, entity_type=etype)
        entity_ids.append(eid)

    # Link entities to objects
    store.add_mention(obj_id=obj_ids[0], entity_id=entity_ids[0])
    store.add_mention(obj_id=obj_ids[3], entity_id=entity_ids[1])
    store.add_mention(obj_id=obj_ids[6], entity_id=entity_ids[2])

    result = gq.project_overview("gamma")

    assert result["project"] == "gamma"
    assert result["object_count"] == 8, f"Expected 8 objects, got {result['object_count']}"

    # All objects present
    result_ids = {obj["id"] for obj in result["objects"]}
    for oid in obj_ids:
        assert oid in result_ids, f"Object {oid} must be in project overview"

    # All entities present
    result_entity_names = {e["name"] for e in result["entities"]}
    for name, _ in entity_names:
        assert name in result_entity_names, f"Entity '{name}' must be in project overview"

    # Edges present (at least the 5 we wired)
    assert len(result["edges"]) >= 5, (
        f"Expected at least 5 edges, got {len(result['edges'])}"
    )


# ── 5. Contradiction map scoping ─────────────────────────────────────


@pytest.mark.bench
def test_contradiction_map_scoping(store: Store) -> None:
    """Contradiction map with scope filters to one project only.

    Seed 2 contradiction pairs in "alpha" and 1 in "beta". Query scoped
    to "alpha" should return exactly 2 pairs with no beta contamination.
    """
    gq = GraphQueries(store)

    # Alpha contradictions: pair 1
    a1 = store.create(obj_type="decision", title="Alpha decision 1", content="Use REST", project="alpha")
    a2 = store.create(obj_type="decision", title="Alpha decision 2", content="Use GraphQL", project="alpha")
    store.create_relationship(from_id=a1, rel_type="contradicts", to_id=a2)

    # Alpha contradictions: pair 2
    a3 = store.create(obj_type="decision", title="Alpha decision 3", content="Monolith", project="alpha")
    a4 = store.create(obj_type="decision", title="Alpha decision 4", content="Microservices", project="alpha")
    store.create_relationship(from_id=a3, rel_type="contradicts", to_id=a4)

    # Beta contradiction: 1 pair
    b1 = store.create(obj_type="decision", title="Beta decision 1", content="SQL", project="beta")
    b2 = store.create(obj_type="decision", title="Beta decision 2", content="NoSQL", project="beta")
    store.create_relationship(from_id=b1, rel_type="contradicts", to_id=b2)

    # Run ReasonStage to generate symmetric contradicts edges
    ReasonStage(store.graph).run()

    result = gq.contradiction_map(scope="alpha")

    assert len(result) == 2, f"Expected exactly 2 alpha pairs, got {len(result)}"

    # Collect all object IDs referenced in the results
    alpha_ids = {a1, a2, a3, a4}
    beta_ids = {b1, b2}

    result_obj_ids: set[str] = set()
    for pair in result:
        result_obj_ids.add(pair["object_a"])
        result_obj_ids.add(pair["object_b"])

    # All result IDs should be from alpha
    assert result_obj_ids <= alpha_ids, (
        f"All contradiction IDs must be from alpha. Got beta contamination: "
        f"{result_obj_ids - alpha_ids}"
    )
    # No beta IDs present
    assert not (result_obj_ids & beta_ids), "No beta objects should appear in alpha-scoped results"
