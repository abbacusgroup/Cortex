"""Benchmark: pattern detection, gap analysis, and staleness propagation.

Tests AdvancedReasoner's ability to detect systemic issues (repeated entity
mentions in fixes), identify knowledge gaps (missing decisions/lessons), and
propagate staleness through dependency chains.
"""

from __future__ import annotations

import datetime

import pytest

from cortex.db.store import Store
from cortex.pipeline.advanced_reason import AdvancedReasoner

from benchmarks.metrics.classification import precision_recall_f1


# ── Helpers ────────────────────────────────────────────────────────


def _old_timestamp(days_ago: int) -> str:
    """Return an ISO-8601 timestamp `days_ago` days in the past."""
    dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_ago)
    return dt.isoformat()


# ── 1. Systemic Issue Detection ───────────────────────────────────


@pytest.mark.bench
def test_systemic_issue_detection(store: Store) -> None:
    """Detect entities with repeated fix mentions (true positives)
    while ignoring below-threshold, outdated, and scattered mentions
    (true negatives).
    """
    reasoner = AdvancedReasoner(store)

    # ── True positives ────────────────────────────────────────

    # Flaky-Service: 4 recent fixes mentioning the same entity
    flaky_eid, _ = store.create_entity(name="Flaky-Service", entity_type="technology")
    for i in range(4):
        fix_id = store.create(
            obj_type="fix",
            title=f"Fix flaky-service issue #{i + 1}",
            content=f"Resolved flaky-service failure variant {i + 1}.",
        )
        store.add_mention(obj_id=fix_id, entity_id=flaky_eid)

    # Unstable-DB: 3 recent fixes (exactly at threshold)
    unstable_eid, _ = store.create_entity(name="Unstable-DB", entity_type="technology")
    for i in range(3):
        fix_id = store.create(
            obj_type="fix",
            title=f"Fix unstable-db issue #{i + 1}",
            content=f"Patched unstable-db problem {i + 1}.",
        )
        store.add_mention(obj_id=fix_id, entity_id=unstable_eid)

    # ── True negatives ────────────────────────────────────────

    # Stable-API: only 2 fixes (below threshold of 3)
    stable_eid, _ = store.create_entity(name="Stable-API", entity_type="technology")
    for i in range(2):
        fix_id = store.create(
            obj_type="fix",
            title=f"Fix stable-api edge case #{i + 1}",
            content=f"Minor stable-api fix {i + 1}.",
        )
        store.add_mention(obj_id=fix_id, entity_id=stable_eid)

    # Old-Bug: 3 fixes but all created 30 days ago (outside 14-day window)
    old_eid, _ = store.create_entity(name="Old-Bug", entity_type="technology")
    old_ts = _old_timestamp(30)
    for i in range(3):
        fix_id = store.create(
            obj_type="fix",
            title=f"Fix old-bug incident #{i + 1}",
            content=f"Historical old-bug fix {i + 1}.",
            created_at=old_ts,
        )
        store.add_mention(obj_id=fix_id, entity_id=old_eid)

    # Scattered-Fix: 3 fixes that each mention a *different* entity
    scattered_eids = []
    for i in range(3):
        eid, _ = store.create_entity(
            name=f"Scattered-Target-{i + 1}", entity_type="technology"
        )
        scattered_eids.append(eid)
        fix_id = store.create(
            obj_type="fix",
            title=f"Fix scattered issue #{i + 1}",
            content=f"One-off fix for scattered target {i + 1}.",
        )
        store.add_mention(obj_id=fix_id, entity_id=eid)

    # ── Run detection ─────────────────────────────────────────

    findings = reasoner.detect_patterns(window_days=14, threshold=3)
    systemic = [f for f in findings if f["type"] == "systemic_issue"]

    detected_eids = {f["entity_id"] for f in systemic}

    # Flaky-Service and Unstable-DB must be detected
    assert flaky_eid in detected_eids, "Flaky-Service should be detected"
    assert unstable_eid in detected_eids, "Unstable-DB should be detected"

    # True negatives must NOT be detected
    assert stable_eid not in detected_eids, "Stable-API (below threshold) must not be detected"
    assert old_eid not in detected_eids, "Old-Bug (outside window) must not be detected"
    for eid in scattered_eids:
        assert eid not in detected_eids, "Scattered entities must not be detected"

    # Verify finding structure for Flaky-Service
    flaky_finding = next(f for f in systemic if f["entity_id"] == flaky_eid)
    assert flaky_finding["entity_name"] == "Flaky-Service"
    assert len(flaky_finding["fix_ids"]) == 4
    assert flaky_finding["window_days"] == 14

    # ── Precision / Recall / F1 ───────────────────────────────

    expected = [
        {"entity_id": flaky_eid},
        {"entity_id": unstable_eid},
    ]

    scores = precision_recall_f1(
        detected=systemic,
        expected=expected,
        match_fn=lambda d, e: d.get("entity_id") == e.get("entity_id"),
    )

    assert scores["precision"] == 1.0, f"Precision should be 1.0, got {scores['precision']}"
    assert scores["recall"] == 1.0, f"Recall should be 1.0, got {scores['recall']}"
    assert scores["f1"] == 1.0, f"F1 should be 1.0, got {scores['f1']}"


# ── 2. Gap Analysis ──────────────────────────────────────────────


@pytest.mark.bench
def test_gap_analysis(store: Store) -> None:
    """Detect missing decisions and uncaptured lessons while ignoring
    projects and entities that have complete coverage.
    """
    reasoner = AdvancedReasoner(store)

    # ── True positives ────────────────────────────────────────

    # Project "gapproject" has sessions but no decisions
    for i in range(3):
        store.create(
            obj_type="session",
            title=f"Gap-project session {i + 1}",
            content=f"Session notes for gapproject #{i + 1}.",
            project="gapproject",
        )

    # Entity "Buggy-Lib" has fixes but no lessons
    buggy_eid, _ = store.create_entity(name="Buggy-Lib", entity_type="technology")
    for i in range(2):
        fix_id = store.create(
            obj_type="fix",
            title=f"Fix buggy-lib crash #{i + 1}",
            content=f"Patched buggy-lib issue {i + 1}.",
        )
        store.add_mention(obj_id=fix_id, entity_id=buggy_eid)

    # ── True negatives ────────────────────────────────────────

    # Project "completeproject" has sessions AND decisions
    for i in range(2):
        store.create(
            obj_type="session",
            title=f"Complete-project session {i + 1}",
            content=f"Session for completeproject #{i + 1}.",
            project="completeproject",
        )
    for i in range(2):
        store.create(
            obj_type="decision",
            title=f"Complete-project decision {i + 1}",
            content=f"Decision for completeproject #{i + 1}.",
            project="completeproject",
        )

    # Entity "Learned-Lib" has fixes AND a lesson
    learned_eid, _ = store.create_entity(name="Learned-Lib", entity_type="technology")
    for i in range(2):
        fix_id = store.create(
            obj_type="fix",
            title=f"Fix learned-lib issue #{i + 1}",
            content=f"Resolved learned-lib problem {i + 1}.",
        )
        store.add_mention(obj_id=fix_id, entity_id=learned_eid)
    lesson_id = store.create(
        obj_type="lesson",
        title="Learned-Lib lesson",
        content="Key takeaway from learned-lib fixes.",
    )
    store.add_mention(obj_id=lesson_id, entity_id=learned_eid)

    # ── Run detection ─────────────────────────────────────────

    findings = reasoner.detect_gaps()

    # Check missing_decisions
    missing_decisions = [f for f in findings if f["type"] == "missing_decisions"]
    gap_projects = {f["project"] for f in missing_decisions}
    assert "gapproject" in gap_projects, "gapproject should trigger missing_decisions"
    assert "completeproject" not in gap_projects, (
        "completeproject should NOT trigger missing_decisions"
    )

    # Check missing_lessons
    missing_lessons = [f for f in findings if f["type"] == "missing_lessons"]
    lesson_eids = {f["entity_id"] for f in missing_lessons}
    assert buggy_eid in lesson_eids, "Buggy-Lib should trigger missing_lessons"
    assert learned_eid not in lesson_eids, (
        "Learned-Lib should NOT trigger missing_lessons"
    )

    # Verify finding structure
    buggy_finding = next(f for f in missing_lessons if f["entity_id"] == buggy_eid)
    assert buggy_finding["entity_name"] == "Buggy-Lib"


# ── 3. Staleness Propagation ─────────────────────────────────────


@pytest.mark.bench
def test_staleness_propagation(store: Store) -> None:
    """Detect objects that depend on superseded objects, and verify
    that non-superseded dependencies are not flagged.
    """
    reasoner = AdvancedReasoner(store)

    # ── Scenario 1: C depends on A, and B supersedes A → C is stale

    a_id = store.create(
        obj_type="decision",
        title="Decision A (superseded)",
        content="Original architecture decision.",
    )
    b_id = store.create(
        obj_type="decision",
        title="Decision B (supersedes A)",
        content="Updated architecture decision replacing A.",
    )
    c_id = store.create(
        obj_type="fix",
        title="Fix C (depends on A)",
        content="Implementation based on decision A.",
    )
    store.create_relationship(from_id=b_id, rel_type="supersedes", to_id=a_id)
    store.create_relationship(from_id=c_id, rel_type="dependsOn", to_id=a_id)

    # ── Scenario 2: E depends on D (not superseded) → E is NOT stale

    d_id = store.create(
        obj_type="decision",
        title="Decision D (current)",
        content="Active decision with no replacement.",
    )
    e_id = store.create(
        obj_type="fix",
        title="Fix E (depends on D)",
        content="Implementation based on current decision D.",
    )
    store.create_relationship(from_id=e_id, rel_type="dependsOn", to_id=d_id)

    # ── Scenario 3: Chain — G supersedes F, H depends on F → H is stale

    f_id = store.create(
        obj_type="decision",
        title="Decision F (superseded)",
        content="Old decision in chain scenario.",
    )
    g_id = store.create(
        obj_type="decision",
        title="Decision G (supersedes F)",
        content="Replacement for decision F.",
    )
    h_id = store.create(
        obj_type="fix",
        title="Fix H (depends on F)",
        content="Work based on superseded decision F.",
    )
    store.create_relationship(from_id=g_id, rel_type="supersedes", to_id=f_id)
    store.create_relationship(from_id=h_id, rel_type="dependsOn", to_id=f_id)

    # ── Run detection ─────────────────────────────────────────

    findings = reasoner.propagate_staleness()
    stale = [f for f in findings if f["type"] == "stale_dependency"]

    stale_obj_ids = {f["object_id"] for f in stale}
    stale_dep_ids = {f["dependency_id"] for f in stale}

    # C and H should be flagged as stale
    assert c_id in stale_obj_ids, "Fix C should be flagged as stale (depends on superseded A)"
    assert h_id in stale_obj_ids, "Fix H should be flagged as stale (depends on superseded F)"

    # E should NOT be flagged
    assert e_id not in stale_obj_ids, (
        "Fix E should NOT be flagged (depends on non-superseded D)"
    )

    # Verify the dependency targets
    assert a_id in stale_dep_ids, "A should appear as a stale dependency target"
    assert f_id in stale_dep_ids, "F should appear as a stale dependency target"
    assert d_id not in stale_dep_ids, "D should NOT appear as a stale dependency target"

    # Verify finding structure for C
    c_finding = next(f for f in stale if f["object_id"] == c_id)
    assert c_finding["dependency_id"] == a_id
