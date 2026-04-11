"""OWL-RL reasoning completeness benchmark.

Exercises the ReasonStage against seven graph topologies, measuring:
- Completeness: did the reasoner produce all expected inferred triples?
- Cycle safety: does the reasoner terminate without creating self-edges?
- Idempotency: does a second run produce zero new triples?
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from cortex.pipeline.reason import ReasonStage

from benchmarks.b2_reasoning.topologies import (
    causal_batch,
    chain,
    contradicts_batch,
    cycle,
    binary_tree,
    diamond,
    star,
)
from benchmarks.metrics.reasoning import completeness_ratio


# ---------------------------------------------------------------------------
# Topology registry
# ---------------------------------------------------------------------------

TOPOLOGIES: list[dict[str, Any]] = [
    {
        "name": "chain_5",
        "builder": lambda s: chain(s, depth=5),
        "expected_key": "expected_transitive",
        "rule_family": "transitive_supersedes_1hop",
        "is_cycle": False,
    },
    {
        "name": "chain_10",
        "builder": lambda s: chain(s, depth=10),
        "expected_key": "expected_transitive",
        "rule_family": "transitive_supersedes_1hop",
        "is_cycle": False,
    },
    {
        "name": "star_8",
        "builder": lambda s: star(s, size=8),
        "expected_key": None,  # no new edges expected
        "rule_family": None,
        "is_cycle": False,
    },
    {
        "name": "binary_tree_4",
        "builder": lambda s: binary_tree(s, depth=4),
        "expected_key": "expected_transitive",
        "rule_family": "transitive_supersedes_1hop",
        "is_cycle": False,
    },
    {
        "name": "diamond",
        "builder": lambda s: diamond(s),
        "expected_key": "expected_transitive",
        "rule_family": "transitive_supersedes_1hop",
        "is_cycle": False,
    },
    {
        "name": "cycle_4",
        "builder": lambda s: cycle(s, size=4),
        "expected_key": None,
        "rule_family": None,
        "is_cycle": True,
    },
    {
        "name": "contradicts_5",
        "builder": lambda s: contradicts_batch(s, size=5),
        "expected_key": "expected_symmetric",
        "rule_family": "symmetric_contradicts",
        "is_cycle": False,
    },
    {
        "name": "causal_5",
        "builder": lambda s: causal_batch(s, size=5),
        "expected_key": "expected_inverse",
        "rule_family": "inverse_causedBy_ledTo",
        "is_cycle": False,
    },
]


def _count_inferred_for_rule(result: dict, rule_family: str | None) -> int:
    """Extract inferred count for a specific rule family from ReasonStage result."""
    if rule_family is None:
        return result.get("total_inferred", 0)
    return result.get("rule_counts", {}).get(rule_family, 0)


def _has_self_edges(store, obj_ids: list[str]) -> bool:
    """Check whether any object has a relationship pointing to itself."""
    for obj_id in obj_ids:
        rels = store.get_relationships(obj_id)
        for rel in rels:
            if rel["other_id"] == obj_id:
                return True
    return False


# ---------------------------------------------------------------------------
# Parametrized benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.bench
@pytest.mark.parametrize(
    "topology",
    TOPOLOGIES,
    ids=[t["name"] for t in TOPOLOGIES],
)
def test_reasoning_bench(store, topology, capsys):
    """Run reasoning on a topology and verify completeness, safety, idempotency."""
    name = topology["name"]
    builder = topology["builder"]
    expected_key = topology["expected_key"]
    rule_family = topology["rule_family"]
    is_cycle = topology["is_cycle"]

    # --- Seed the graph ---
    t0 = time.perf_counter()
    info = builder(store)
    seed_ms = (time.perf_counter() - t0) * 1000

    # Determine expected inferred count
    if expected_key is not None:
        expected = info[expected_key]
    else:
        expected = 0

    # --- First reasoning pass ---
    reasoner = ReasonStage(store.graph)
    t0 = time.perf_counter()
    result = reasoner.run()
    reason_ms = (time.perf_counter() - t0) * 1000

    inferred = _count_inferred_for_rule(result, rule_family)

    # --- Completeness ---
    if not is_cycle:
        ratio = completeness_ratio(inferred, expected)
        assert ratio == 1.0, (
            f"[{name}] completeness {ratio:.3f} != 1.0 "
            f"(inferred={inferred}, expected={expected})"
        )

    # --- Cycle safety ---
    if is_cycle:
        obj_ids = info.get("obj_ids", [])
        assert not _has_self_edges(store, obj_ids), (
            f"[{name}] self-edge detected after reasoning on cycle"
        )

    # --- Idempotency (second pass should infer nothing) ---
    t0 = time.perf_counter()
    result2 = reasoner.run()
    idempotent_ms = (time.perf_counter() - t0) * 1000
    new_triples = result2.get("total_inferred", 0)
    assert new_triples == 0, (
        f"[{name}] second reasoning pass inferred {new_triples} triples (expected 0)"
    )

    # --- Results table ---
    with capsys.disabled():
        print(
            f"\n{'='*60}\n"
            f"Topology:     {name}\n"
            f"Expected:     {expected}\n"
            f"Inferred:     {inferred}\n"
            f"Completeness: {completeness_ratio(inferred, expected):.3f}\n"
            f"Idempotent:   {new_triples == 0}\n"
            f"Iterations:   {result.get('iterations', '?')}\n"
            f"Seed time:    {seed_ms:.1f} ms\n"
            f"Reason time:  {reason_ms:.1f} ms\n"
            f"Idem. time:   {idempotent_ms:.1f} ms\n"
            f"{'='*60}"
        )
