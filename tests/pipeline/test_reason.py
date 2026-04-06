"""Tests for the ReasonStage pipeline stage (OWL-RL inference)."""

from __future__ import annotations

import pytest

from cortex.db.graph_store import GraphStore
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.reason import ReasonStage


@pytest.fixture
def graph():
    g = GraphStore(path=None)  # in-memory
    g.load_ontology(find_ontology())
    return g


@pytest.fixture
def reasoner(graph):
    return ReasonStage(graph)


def _create_pair(graph, title_a="A", title_b="B"):
    """Create two decision objects and return their IDs."""
    id_a = graph.create_object(obj_type="decision", title=title_a)
    id_b = graph.create_object(obj_type="decision", title=title_b)
    return id_a, id_b


class TestSymmetry:
    """contradicts is symmetric: A contradicts B -> B contradicts A."""

    def test_symmetric_contradicts(self, graph, reasoner):
        id_a, id_b = _create_pair(graph)
        graph.create_relationship(
            from_id=id_a, rel_type="contradicts", to_id=id_b
        )

        result = reasoner.run()

        assert result["total_inferred"] >= 1
        assert result["rule_counts"].get("symmetric_contradicts", 0) >= 1

        # Verify the reverse triple exists
        rels_b = graph.get_relationships(id_b)
        outgoing = [
            r for r in rels_b
            if r["direction"] == "outgoing"
            and r["rel_type"] == "contradicts"
        ]
        assert any(r["other_id"] == id_a for r in outgoing)


class TestInverse:
    """causedBy <-> ledTo are inverses."""

    def test_caused_by_infers_led_to(self, graph, reasoner):
        id_a, id_b = _create_pair(graph)
        graph.create_relationship(
            from_id=id_a, rel_type="causedBy", to_id=id_b
        )

        result = reasoner.run()

        assert result["rule_counts"].get("inverse_causedBy_ledTo", 0) >= 1
        rels_b = graph.get_relationships(id_b)
        outgoing_led = [
            r for r in rels_b
            if r["direction"] == "outgoing" and r["rel_type"] == "ledTo"
        ]
        assert any(r["other_id"] == id_a for r in outgoing_led)

    def test_led_to_infers_caused_by(self, graph, reasoner):
        id_a, id_b = _create_pair(graph)
        graph.create_relationship(
            from_id=id_a, rel_type="ledTo", to_id=id_b
        )

        result = reasoner.run()

        assert result["rule_counts"].get("inverse_ledTo_causedBy", 0) >= 1
        rels_b = graph.get_relationships(id_b)
        outgoing_caused = [
            r for r in rels_b
            if r["direction"] == "outgoing"
            and r["rel_type"] == "causedBy"
        ]
        assert any(r["other_id"] == id_a for r in outgoing_caused)


class TestTransitivity:
    """supersedes is transitive: A->B->C implies A->C."""

    def test_one_hop_transitivity(self, graph, reasoner):
        id_a, id_b = _create_pair(graph, "A", "B")
        id_c = graph.create_object(obj_type="decision", title="C")

        graph.create_relationship(
            from_id=id_a, rel_type="supersedes", to_id=id_b
        )
        graph.create_relationship(
            from_id=id_b, rel_type="supersedes", to_id=id_c
        )

        result = reasoner.run()

        assert (
            result["rule_counts"].get("transitive_supersedes_1hop", 0) >= 1
        )
        rels_a = graph.get_relationships(id_a)
        supersedes_targets = [
            r["other_id"]
            for r in rels_a
            if r["direction"] == "outgoing"
            and r["rel_type"] == "supersedes"
        ]
        assert id_c in supersedes_targets

    def test_ten_deep_chain(self, graph, reasoner):
        """10-deep supersedes chain -> all transitive closures computed."""
        ids = []
        for i in range(11):
            obj_id = graph.create_object(
                obj_type="decision", title=f"Node-{i}"
            )
            ids.append(obj_id)

        for i in range(10):
            graph.create_relationship(
                from_id=ids[i], rel_type="supersedes", to_id=ids[i + 1]
            )

        reasoner.run()

        # The first node should transitively supersede all later nodes
        rels_first = graph.get_relationships(ids[0])
        supersedes_targets = {
            r["other_id"]
            for r in rels_first
            if r["direction"] == "outgoing"
            and r["rel_type"] == "supersedes"
        }
        for later_id in ids[1:]:
            assert later_id in supersedes_targets, (
                f"Missing transitive supersedes from {ids[0][:8]} "
                f"to {later_id[:8]}"
            )

    def test_circular_transitivity_terminates(self, graph, reasoner):
        """A->B->C->A cycle terminates without infinite loop."""
        id_a = graph.create_object(obj_type="decision", title="CycA")
        id_b = graph.create_object(obj_type="decision", title="CycB")
        id_c = graph.create_object(obj_type="decision", title="CycC")

        graph.create_relationship(
            from_id=id_a, rel_type="supersedes", to_id=id_b
        )
        graph.create_relationship(
            from_id=id_b, rel_type="supersedes", to_id=id_c
        )
        graph.create_relationship(
            from_id=id_c, rel_type="supersedes", to_id=id_a
        )

        # Should not hang — FILTER(?a != ?c) prevents self-supersedes
        result = reasoner.run(max_iterations=20)
        assert result["iterations"] <= 20


class TestFixpoint:
    """Fixpoint and idempotency tests."""

    def test_idempotent_on_rerun(self, graph, reasoner):
        """Second run produces zero new triples."""
        id_a, id_b = _create_pair(graph)
        graph.create_relationship(
            from_id=id_a, rel_type="contradicts", to_id=id_b
        )

        first = reasoner.run()
        assert first["total_inferred"] >= 1

        second = reasoner.run()
        assert second["total_inferred"] == 0

    def test_no_applicable_rules(self, graph, reasoner):
        """Graph with no relationships -> total_inferred=0, iterations=1."""
        graph.create_object(obj_type="idea", title="Lonely")

        result = reasoner.run()

        assert result["total_inferred"] == 0
        assert result["iterations"] == 1

    def test_fixpoint_before_max_iterations(self, graph, reasoner):
        """Simple case reaches fixpoint well before max_iterations."""
        id_a, id_b = _create_pair(graph)
        graph.create_relationship(
            from_id=id_a, rel_type="contradicts", to_id=id_b
        )

        result = reasoner.run(max_iterations=50)

        assert result["iterations"] < 50
