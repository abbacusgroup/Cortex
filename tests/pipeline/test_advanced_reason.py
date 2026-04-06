"""Tests for AdvancedReasoner — contradiction, pattern, gap, staleness, causal."""

from __future__ import annotations

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.advanced_reason import AdvancedReasoner


@pytest.fixture
def store(tmp_path):
    config = CortexConfig(data_dir=tmp_path)
    s = Store(config)
    s.initialize(find_ontology())
    return s


@pytest.fixture
def reasoner(store):
    return AdvancedReasoner(store)


# ── helpers ──────────────────────────────────────────────────────────


def _create(store, obj_type="decision", title="obj", **kw):
    """Create a knowledge object and return its ID."""
    return store.create(obj_type=obj_type, title=title, **kw)


# ── Contradiction Detection ──────────────────────────────────────────


class TestContradictions:
    """Superseded object with/without active dependents."""

    def test_superseded_with_dependents(self, store, reasoner):
        """Superseded obj + incoming dependsOn -> contradiction."""
        old = _create(store, title="Old decision")
        new = _create(store, title="New decision")
        dep = _create(store, title="Dependent")

        # new supersedes old
        store.create_relationship(from_id=new, rel_type="supersedes", to_id=old)
        # dep depends on old (still)
        store.create_relationship(from_id=dep, rel_type="dependsOn", to_id=old)

        result = reasoner.detect_contradictions()

        assert len(result) == 1
        finding = result[0]
        assert finding["type"] == "structural_contradiction"
        assert finding["severity"] == "high"
        assert finding["object_id"] == old
        assert dep in finding["dependent_ids"]

    def test_superseded_without_dependents(self, store, reasoner):
        """Superseded obj + no dependents -> no contradiction."""
        old = _create(store, title="Old decision")
        new = _create(store, title="New decision")

        store.create_relationship(from_id=new, rel_type="supersedes", to_id=old)

        result = reasoner.detect_contradictions()
        assert result == []

    def test_not_superseded(self, store, reasoner):
        """Non-superseded obj with dependents -> no contradiction."""
        obj = _create(store, title="Active decision")
        dep = _create(store, title="Dependent")

        store.create_relationship(from_id=dep, rel_type="dependsOn", to_id=obj)

        result = reasoner.detect_contradictions()
        assert result == []

    def test_empty_graph(self, store, reasoner):
        """No objects at all -> empty list."""
        result = reasoner.detect_contradictions()
        assert result == []


# ── Pattern Detection ────────────────────────────────────────────────


class TestPatterns:
    """3+ fixes mentioning same entity within window -> systemic issue."""

    def test_three_fixes_same_entity_detected(self, store, reasoner):
        """3 recent fixes mentioning one entity -> systemic_issue."""
        entity_id = store.create_entity(name="AuthService", entity_type="technology")
        for i in range(3):
            fix_id = _create(
                store,
                obj_type="fix",
                title=f"Fix auth {i}",
                project="cortex",
            )
            store.add_mention(obj_id=fix_id, entity_id=entity_id)

        result = reasoner.detect_patterns(window_days=14, threshold=3)

        assert len(result) == 1
        assert result[0]["type"] == "systemic_issue"
        assert result[0]["entity_name"] == "AuthService"
        assert len(result[0]["fix_ids"]) == 3

    def test_below_threshold_not_detected(self, store, reasoner):
        """Only 2 fixes (below threshold=3) -> no pattern."""
        entity_id = store.create_entity(name="CacheLayer", entity_type="technology")
        for i in range(2):
            fix_id = _create(
                store,
                obj_type="fix",
                title=f"Fix cache {i}",
                project="cortex",
            )
            store.add_mention(obj_id=fix_id, entity_id=entity_id)

        result = reasoner.detect_patterns(window_days=14, threshold=3)
        assert result == []

    def test_old_fixes_outside_window(self, store, reasoner):
        """3 fixes but older than window -> not detected."""
        entity_id = store.create_entity(name="OldModule", entity_type="technology")
        for i in range(3):
            fix_id = _create(
                store,
                obj_type="fix",
                title=f"Old fix {i}",
                project="cortex",
            )
            store.add_mention(obj_id=fix_id, entity_id=entity_id)

        # Use a 0-day window so everything is "old"
        result = reasoner.detect_patterns(window_days=0, threshold=3)
        assert result == []


# ── Gap Analysis ─────────────────────────────────────────────────────


class TestGaps:
    """Missing decisions for projects; missing lessons for entities."""

    def test_sessions_without_decisions(self, store, reasoner):
        """Project with sessions but no decisions -> missing_decisions."""
        _create(
            store,
            obj_type="session",
            title="Sprint planning",
            project="acme",
        )

        result = reasoner.detect_gaps()

        decision_gaps = [g for g in result if g["type"] == "missing_decisions"]
        assert len(decision_gaps) == 1
        assert decision_gaps[0]["project"] == "acme"

    def test_sessions_with_decisions_no_gap(self, store, reasoner):
        """Project with sessions AND decisions -> no gap."""
        _create(
            store,
            obj_type="session",
            title="Sprint planning",
            project="acme",
        )
        _create(
            store,
            obj_type="decision",
            title="Use Postgres",
            project="acme",
        )

        result = reasoner.detect_gaps()

        decision_gaps = [g for g in result if g["type"] == "missing_decisions"]
        assert decision_gaps == []

    def test_fixes_without_lessons(self, store, reasoner):
        """Entity with fixes but no lessons -> missing_lessons."""
        entity_id = store.create_entity(name="DBPool", entity_type="technology")
        fix_id = _create(store, obj_type="fix", title="Fix pool leak")
        store.add_mention(obj_id=fix_id, entity_id=entity_id)

        result = reasoner.detect_gaps()

        lesson_gaps = [g for g in result if g["type"] == "missing_lessons"]
        assert len(lesson_gaps) == 1
        assert lesson_gaps[0]["entity_name"] == "DBPool"

    def test_fixes_with_lessons_no_gap(self, store, reasoner):
        """Entity with both fixes and lessons -> no gap."""
        entity_id = store.create_entity(name="DBPool", entity_type="technology")
        fix_id = _create(store, obj_type="fix", title="Fix pool leak")
        lesson_id = _create(store, obj_type="lesson", title="Pool sizing matters")
        store.add_mention(obj_id=fix_id, entity_id=entity_id)
        store.add_mention(obj_id=lesson_id, entity_id=entity_id)

        result = reasoner.detect_gaps()

        lesson_gaps = [g for g in result if g["type"] == "missing_lessons"]
        assert lesson_gaps == []


# ── Staleness Propagation ────────────────────────────────────────────


class TestStaleness:
    """Objects depending on superseded objects -> stale_dependency."""

    def test_depends_on_superseded(self, store, reasoner):
        """B depends on A, A is superseded -> stale_dependency."""
        old = _create(store, title="Old design")
        new = _create(store, title="New design")
        consumer = _create(store, title="Consumer")

        store.create_relationship(from_id=new, rel_type="supersedes", to_id=old)
        store.create_relationship(from_id=consumer, rel_type="dependsOn", to_id=old)

        result = reasoner.propagate_staleness()

        assert len(result) == 1
        assert result[0]["type"] == "stale_dependency"
        assert result[0]["object_id"] == consumer
        assert result[0]["dependency_id"] == old

    def test_depends_on_active(self, store, reasoner):
        """B depends on A, A is NOT superseded -> no staleness."""
        active = _create(store, title="Active design")
        consumer = _create(store, title="Consumer")

        store.create_relationship(from_id=consumer, rel_type="dependsOn", to_id=active)

        result = reasoner.propagate_staleness()
        assert result == []


# ── Causal Chain Assembly ────────────────────────────────────────────


class TestCausalChain:
    """Follow causedBy/ledTo edges to build narrative."""

    def test_caused_by_chain(self, store, reasoner):
        """A causedBy B -> chain includes both."""
        bug = _create(store, obj_type="fix", title="Bug")
        decision = _create(store, obj_type="decision", title="Design choice")

        store.create_relationship(from_id=bug, rel_type="causedBy", to_id=decision)

        chain = reasoner.assemble_causal_chain(bug)

        chain_ids = [entry["id"] for entry in chain]
        assert bug in chain_ids
        assert decision in chain_ids

    def test_no_causal_relations(self, store, reasoner):
        """Object with no causal edges -> chain has just itself."""
        solo = _create(store, obj_type="idea", title="Standalone idea")

        chain = reasoner.assemble_causal_chain(solo)

        assert len(chain) == 1
        assert chain[0]["id"] == solo
        assert chain[0]["title"] == "Standalone idea"

    def test_multi_hop_chain(self, store, reasoner):
        """decision -> bug -> fix via causedBy edges."""
        decision = _create(store, obj_type="decision", title="API design")
        bug = _create(store, obj_type="fix", title="Auth bug")
        fix = _create(store, obj_type="fix", title="Auth fix")

        store.create_relationship(from_id=bug, rel_type="causedBy", to_id=decision)
        store.create_relationship(from_id=fix, rel_type="causedBy", to_id=bug)

        chain = reasoner.assemble_causal_chain(fix)

        chain_ids = [e["id"] for e in chain]
        assert decision in chain_ids
        assert bug in chain_ids
        assert fix in chain_ids


# ── run_all ──────────────────────────────────────────────────────────


class TestRunAll:
    """run_all returns dict with all 4 categories."""

    def test_returns_all_categories(self, store, reasoner):
        result = reasoner.run_all()

        assert "contradictions" in result
        assert "patterns" in result
        assert "gaps" in result
        assert "staleness" in result
        assert isinstance(result["contradictions"], list)
        assert isinstance(result["patterns"], list)
        assert isinstance(result["gaps"], list)
        assert isinstance(result["staleness"], list)

    def test_populated_graph(self, store, reasoner):
        """run_all aggregates findings from all sub-checks."""
        # Set up a contradiction
        old = _create(store, title="Old")
        new = _create(store, title="New")
        dep = _create(store, title="Dep")
        store.create_relationship(from_id=new, rel_type="supersedes", to_id=old)
        store.create_relationship(from_id=dep, rel_type="dependsOn", to_id=old)

        # Set up a gap
        _create(
            store,
            obj_type="session",
            title="Planning",
            project="gapproj",
        )

        result = reasoner.run_all()

        assert len(result["contradictions"]) >= 1
        assert len(result["staleness"]) >= 1
        gap_projs = [g["project"] for g in result["gaps"] if g["type"] == "missing_decisions"]
        assert "gapproj" in gap_projs
