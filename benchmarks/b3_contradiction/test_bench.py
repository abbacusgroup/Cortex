"""B3 — Structural contradiction detection benchmark.

Tests that AdvancedReasoner.detect_contradictions() correctly identifies
objects that are superseded yet still have active dependents, while ignoring
objects that are clean (no contradiction).
"""

from __future__ import annotations

import pytest

from cortex.pipeline.advanced_reason import AdvancedReasoner
from benchmarks.metrics.classification import precision_recall_f1


# ── Helpers ─────────────────────────────────────────────────────────


def _obj(store, title: str, obj_type: str = "decision") -> str:
    """Create an object and return its id."""
    return store.create(obj_type=obj_type, title=title, content=f"Content for {title}", summary=f"Summary for {title}")


def _supersedes(store, newer_id: str, older_id: str) -> None:
    """Record that *newer* supersedes *older*."""
    store.create_relationship(from_id=newer_id, rel_type="supersedes", to_id=older_id)


def _depends_on(store, dependent_id: str, dependency_id: str) -> None:
    """Record that *dependent* depends on *dependency*."""
    store.create_relationship(from_id=dependent_id, rel_type="dependsOn", to_id=dependency_id)


# ── True-Positive scenarios ─────────────────────────────────────────


def _tp_simple(store) -> str:
    """Simple: A superseded by B, C depends on A."""
    a = _obj(store, "TP1-A: Old design")
    b = _obj(store, "TP1-B: New design")
    c = _obj(store, "TP1-C: Consumer of A")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    return a


def _tp_multi_dependent(store) -> str:
    """Multi-dependent: A superseded by B, C/D/E all depend on A."""
    a = _obj(store, "TP2-A: Legacy API spec")
    b = _obj(store, "TP2-B: Updated API spec")
    c = _obj(store, "TP2-C: Service alpha")
    d = _obj(store, "TP2-D: Service beta")
    e = _obj(store, "TP2-E: Service gamma")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    _depends_on(store, d, a)
    _depends_on(store, e, a)
    return a


def _tp_chain(store) -> str:
    """Chain: B supersedes A, C depends on A, D supersedes C.

    A is superseded with dependent C (even though C is also superseded).
    """
    a = _obj(store, "TP3-A: Original decision")
    b = _obj(store, "TP3-B: Revised decision")
    c = _obj(store, "TP3-C: Intermediate consumer")
    d = _obj(store, "TP3-D: Replacement consumer")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    _supersedes(store, d, c)
    return a


def _tp_different_types(store) -> str:
    """Cross-type: fix supersedes decision, lesson depends on decision."""
    a = _obj(store, "TP4-A: Architecture decision", obj_type="decision")
    b = _obj(store, "TP4-B: Hot-fix override", obj_type="fix")
    c = _obj(store, "TP4-C: Lesson referencing A", obj_type="lesson")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    return a


def _tp_two_dependents(store) -> str:
    """Two dependents on a superseded object."""
    a = _obj(store, "TP5-A: Old config")
    b = _obj(store, "TP5-B: New config")
    c = _obj(store, "TP5-C: Dep 1")
    d = _obj(store, "TP5-D: Dep 2")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    _depends_on(store, d, a)
    return a


def _tp_four_dependents(store) -> str:
    """Four dependents on a superseded object."""
    a = _obj(store, "TP6-A: Retired schema")
    b = _obj(store, "TP6-B: Active schema")
    deps = []
    for i in range(4):
        deps.append(_obj(store, f"TP6-Dep{i}: Consumer {i}"))
    _supersedes(store, b, a)
    for d in deps:
        _depends_on(store, d, a)
    return a


def _tp_superseded_fix(store) -> str:
    """Fix type superseded with a dependent lesson."""
    a = _obj(store, "TP7-A: Patch v1", obj_type="fix")
    b = _obj(store, "TP7-B: Patch v2", obj_type="fix")
    c = _obj(store, "TP7-C: Lesson from patch", obj_type="lesson")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    return a


def _tp_single_dependent_lesson(store) -> str:
    """Single dependent lesson on superseded decision."""
    a = _obj(store, "TP8-A: Strategy v1")
    b = _obj(store, "TP8-B: Strategy v2")
    c = _obj(store, "TP8-C: Retrospective", obj_type="lesson")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    return a


def _tp_multiple_supersessions(store) -> str:
    """Object superseded by one, with a dependent that also has its own superseder.

    A superseded by B, C depends on A, C superseded by D, E depends on C.
    Both A and C should be flagged.
    """
    a = _obj(store, "TP9-A: Root doc")
    b = _obj(store, "TP9-B: Replaces root")
    c = _obj(store, "TP9-C: Mid-layer")
    d = _obj(store, "TP9-D: Replaces mid")
    e = _obj(store, "TP9-E: Leaf consumer")
    _supersedes(store, b, a)
    _depends_on(store, c, a)
    _supersedes(store, d, c)
    _depends_on(store, e, c)
    # Both a and c are contradictions
    return a, c  # type: ignore[return-value]


# ── True-Negative scenarios ─────────────────────────────────────────


def _tn_clean_retirement(store) -> None:
    """B supersedes A, but nothing depends on A."""
    a = _obj(store, "TN1-A: Retired cleanly")
    b = _obj(store, "TN1-B: Replacement")
    _supersedes(store, b, a)


def _tn_healthy_dependency(store) -> None:
    """C depends on A, but A is NOT superseded."""
    a = _obj(store, "TN2-A: Active baseline")
    c = _obj(store, "TN2-C: Consumer")
    _depends_on(store, c, a)


def _tn_isolated(store) -> None:
    """Object with no relationships at all."""
    _obj(store, "TN3-A: Standalone note")


def _tn_superseder_not_superseded(store) -> None:
    """A supersedes B — B is superseded but nothing depends on B."""
    a = _obj(store, "TN4-A: Newer")
    b = _obj(store, "TN4-B: Older, no deps")
    _supersedes(store, a, b)


def _tn_only_outgoing_depends(store) -> None:
    """A depends on B (outgoing), A is not superseded."""
    a = _obj(store, "TN5-A: Depends outward")
    b = _obj(store, "TN5-B: Dependency target")
    _depends_on(store, a, b)


def _tn_two_superseded_no_deps(store) -> None:
    """Two objects superseded, neither has dependents."""
    a = _obj(store, "TN6-A: Old v1")
    b = _obj(store, "TN6-B: Old v2")
    c = _obj(store, "TN6-C: Current v3")
    _supersedes(store, b, a)
    _supersedes(store, c, b)


def _tn_depends_chain_no_supersession(store) -> None:
    """A -> B -> C dependency chain, nobody superseded."""
    a = _obj(store, "TN7-A: Base")
    b = _obj(store, "TN7-B: Middle")
    c = _obj(store, "TN7-C: Top")
    _depends_on(store, b, a)
    _depends_on(store, c, b)


def _tn_superseded_depends_outward(store) -> None:
    """A is superseded by B. A itself depends on C (outgoing). No incoming dependsOn on A."""
    a = _obj(store, "TN8-A: Superseded with outgoing dep")
    b = _obj(store, "TN8-B: Replacement")
    c = _obj(store, "TN8-C: Upstream")
    _supersedes(store, b, a)
    _depends_on(store, a, c)


def _tn_multiple_isolated(store) -> None:
    """Several isolated objects, no edges."""
    for i in range(3):
        _obj(store, f"TN9-Obj{i}: Standalone {i}")


def _tn_mutual_depends(store) -> None:
    """Two objects depend on each other, neither superseded."""
    a = _obj(store, "TN10-A: Peer left")
    b = _obj(store, "TN10-B: Peer right")
    _depends_on(store, a, b)
    _depends_on(store, b, a)


# ── Benchmark Test ──────────────────────────────────────────────────


@pytest.mark.bench
def test_contradiction_bench(store) -> None:
    """Structural contradiction detection — precision, recall, F1."""

    # ── Seed true-positive scenarios ──
    expected_ids: list[str] = []
    expected_ids.append(_tp_simple(store))
    expected_ids.append(_tp_multi_dependent(store))
    expected_ids.append(_tp_chain(store))
    expected_ids.append(_tp_different_types(store))
    expected_ids.append(_tp_two_dependents(store))
    expected_ids.append(_tp_four_dependents(store))
    expected_ids.append(_tp_superseded_fix(store))
    expected_ids.append(_tp_single_dependent_lesson(store))

    # TP9 returns two contradicted ids
    tp9_a, tp9_c = _tp_multiple_supersessions(store)
    expected_ids.append(tp9_a)
    expected_ids.append(tp9_c)

    # ── Seed true-negative scenarios (should NOT appear) ──
    _tn_clean_retirement(store)
    _tn_healthy_dependency(store)
    _tn_isolated(store)
    _tn_superseder_not_superseded(store)
    _tn_only_outgoing_depends(store)
    _tn_two_superseded_no_deps(store)
    _tn_depends_chain_no_supersession(store)
    _tn_superseded_depends_outward(store)
    _tn_multiple_isolated(store)
    _tn_mutual_depends(store)

    # ── Run detection ──
    reasoner = AdvancedReasoner(store)
    findings = reasoner.detect_contradictions()

    # ── Evaluate ──
    expected = [{"object_id": oid} for oid in expected_ids]
    scores = precision_recall_f1(findings, expected)

    print(f"\n  Structural contradiction detection")
    print(f"    True positives expected : {len(expected_ids)}")
    print(f"    Findings returned       : {len(findings)}")
    print(f"    Precision               : {scores['precision']:.3f}")
    print(f"    Recall                  : {scores['recall']:.3f}")
    print(f"    F1                      : {scores['f1']:.3f}")

    assert scores["f1"] == 1.0, (
        f"Expected perfect F1 but got {scores['f1']:.3f} "
        f"(P={scores['precision']:.3f}, R={scores['recall']:.3f})"
    )
